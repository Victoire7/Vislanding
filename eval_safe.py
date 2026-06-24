dependencies = ['torch', 'torchvision']

import os
import json
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
from training.mono.utils.seg_metric import IoUMetric

metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
    'ViT-Small': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py',
        'ckpt_file': 'weight/metric_depth_vit_small_800k.pth',
    }
}


def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    See hubconf.py for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(torch.load(ckpt_file)['model_state_dict'], strict=False)
    return model


def metric3d_vit_small_(pretrain, ckpt_file, cfg):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    See hubconf.py for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    # cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    # ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']
    #
    # cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(torch.load(ckpt_file)['model_state_dict'], strict=False)
    return model


def prepare_input(rgb_file, intrinsic):
    rgb_origin_ = cv2.imread(rgb_file)
    H, W, _ = rgb_origin_.shape
    rgb_origin = rgb_origin_[:, :, ::-1]

    input_size = (616, 1064)  # for vit model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                             value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    return rgb, rgb_origin, intrinsic, pad_info


def de_depth(pred_depth, intrinsic, pad_info):
    pred_depth = pred_depth.squeeze()
    # a = pad_info[0]
    # b = pred_depth.shape[0] - pad_info[1]
    # c = pad_info[2]
    # d = pred_depth.shape[1] - pad_info[3]
    pre_deppth_c = pred_depth.cpu().numpy()

    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                 pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    pre_deppth_c_unpad = pred_depth.cpu().numpy()

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2],
                                                 mode='bilinear').squeeze()
    ###################### canonical camera space ######################
    max_pre_d_p = torch.max(pred_depth)
    min_pre_d_p = torch.min(pred_depth)
    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    max_pre_d = torch.max(pred_depth)
    min_pre_d = torch.min(pred_depth)
    pred_depth = torch.clamp(pred_depth, 10, 300)
    return pred_depth


def de_normal(output_dict, pad_info, rgb_origin):
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    normal_confidence = output_dict['prediction_normal'][:, 3, :,
                        :]  # see https://arxiv.org/abs/2109.09881 for details
    # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
                  pad_info[2]: pred_normal.shape[2] - pad_info[3]]
    # you can now do anything with the normal
    pred_normal = torch.nn.functional.interpolate(pred_normal[None, :, :], rgb_origin.shape[:2],
                                                  mode='bilinear').squeeze()
    # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    return pred_normal, pred_normal_vis


def load_safe_gt(path):
    image = cv2.imread(path)
    safe_color = np.array([0, 0, 0])  # safe 颜色
    unsafe_colors = [
        np.array([0, 0, 128]),  # unsafe 颜色 1
        np.array([0, 128, 0]),  # unsafe 颜色 2
    ]

    binary_gt = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    safe_mask = np.all(image == safe_color, axis=-1)
    binary_gt[safe_mask] = 0
    binary_gt[~safe_mask] = 1
    return binary_gt


def eval_wilduav(model):
    seg_metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'],
                           class_names=['safe', 'unsafe'],
                           ignore_index=2)

    dataset = json.load(open("/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/wild_labled_val_4xr_safe.json"))
    intrinsic = [4545.239894100246 / 4, 4545.239894100246 / 4, 2643.5350965445928 / 4, 1965.6392125747034 / 4]
    for data in tqdm.tqdm(dataset['files']):
        rgb_file = data['rgb']
        rgb, rgb_origin, intrinsic, pad_info = prepare_input(rgb_file, intrinsic)
        safe_gt_path = data['safe_ann']
        safe_gt = load_safe_gt(safe_gt_path)

        ###################### canonical camera space ######################
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})

        safe_pre_logits = output_dict['safe_prediction'][0]

        safe_pre_logits_copy = copy.deepcopy(safe_pre_logits)

        safe_pre_logits = safe_pre_logits[:, pad_info[0]: safe_pre_logits.shape[1] - pad_info[1],
                          pad_info[2]: safe_pre_logits.shape[2] - pad_info[3]]
        # you can now do anything with the normal
        safe_pre_logits = torch.nn.functional.interpolate(safe_pre_logits[None, :, :], rgb_origin.shape[:2],
                                                          mode='bilinear').squeeze()
        probs = F.softmax(safe_pre_logits, dim=0)  # 形状 (B, C, H, W)
        pred_labels = torch.argmax(probs, dim=0).cpu().numpy()

        # probs_2 = F.softmax(safe_pre_logits_copy, dim=0)
        # pred_labels_2 = torch.argmax(probs_2, dim=0).cpu()
        # pred_labels_2 = pred_labels_2[pad_info[0]: pred_labels_2.shape[0] - pad_info[1],
        #                   pad_info[2]: pred_labels_2.shape[1] - pad_info[3]].numpy()
        # pred_labels_2 = cv2.resize(pred_labels_2, (rgb_origin.shape[1], rgb_origin.shape[0]), interpolation=cv2.INTER_NEAREST)

        seg_metric.process(torch.tensor(pred_labels), torch.tensor(safe_gt))
    seg_metrics, print_str = seg_metric.compute_metrics(seg_metric.results)
    print(print_str)
    print(seg_metrics)
    return seg_metrics, print_str


def eval_sd(model):
    seg_metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'],
                           class_names=['safe', 'unsafe'],
                           ignore_index=2)

    # dataset = json.load(open("/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/wild_labled_val_4xr_safe.json"))
    intrinsic = [4545.239894100246 / 6, 4545.239894100246 / 6, 750, 500]
    root_dir = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/Semantic_Drone_Dataset/dataset"
    img_dir = os.path.join(root_dir, "rgb_compress")
    ann_dir = os.path.join(root_dir, "safe_ann")
    for img in tqdm.tqdm(os.listdir(img_dir)):
        rgb_file = os.path.join(img_dir, img)
        rgb, rgb_origin, intrinsic, pad_info = prepare_input(rgb_file, intrinsic)
        safe_gt_path = os.path.join(ann_dir, img.replace('.jpg', '.png'))
        safe_gt = load_safe_gt(safe_gt_path)

        ###################### canonical camera space ######################
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})

        safe_pre_logits = output_dict['safe_prediction'][0]

        safe_pre_logits_copy = copy.deepcopy(safe_pre_logits)

        safe_pre_logits = safe_pre_logits[:, pad_info[0]: safe_pre_logits.shape[1] - pad_info[1],
                          pad_info[2]: safe_pre_logits.shape[2] - pad_info[3]]
        # you can now do anything with the normal
        safe_pre_logits = torch.nn.functional.interpolate(safe_pre_logits[None, :, :], rgb_origin.shape[:2],
                                                          mode='bilinear').squeeze()
        probs = F.softmax(safe_pre_logits, dim=0)  # 形状 (B, C, H, W)
        pred_labels = torch.argmax(probs, dim=0).cpu().numpy()

        # probs_2 = F.softmax(safe_pre_logits_copy, dim=0)
        # pred_labels_2 = torch.argmax(probs_2, dim=0).cpu()
        # pred_labels_2 = pred_labels_2[pad_info[0]: pred_labels_2.shape[0] - pad_info[1],
        #                   pad_info[2]: pred_labels_2.shape[1] - pad_info[3]].numpy()
        # pred_labels_2 = cv2.resize(pred_labels_2, (rgb_origin.shape[1], rgb_origin.shape[0]), interpolation=cv2.INTER_NEAREST)

        seg_metric.process(torch.tensor(pred_labels), torch.tensor(safe_gt))
    seg_metrics, print_str = seg_metric.compute_metrics(seg_metric.results)
    print(print_str)
    print(seg_metrics)
    return seg_metrics, print_str

def eval_sd_single(model):


    # dataset = json.load(open("/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/wild_labled_val_4xr_safe.json"))
    intrinsic = [4545.239894100246 / 4, 4545.239894100246 / 4, 2643.5350965445928 / 4, 1965.6392125747034 / 4]
    root_dir = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/Semantic_Drone_Dataset/dataset"
    img_dir = os.path.join(root_dir, "rgb_compress")
    ann_dir = os.path.join(root_dir, "safe_ann")
    results_dict = {}
    for img in tqdm.tqdm(os.listdir(img_dir)):
        seg_metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'],
                               class_names=['safe', 'unsafe'],
                               ignore_index=2)
        rgb_file = os.path.join(img_dir, img)
        rgb, rgb_origin, intrinsic, pad_info = prepare_input(rgb_file, intrinsic)
        safe_gt_path = os.path.join(ann_dir, img.replace('.jpg', '.png'))
        safe_gt = load_safe_gt(safe_gt_path)

        ###################### canonical camera space ######################
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})

        safe_pre_logits = output_dict['safe_prediction'][0]

        safe_pre_logits_copy = copy.deepcopy(safe_pre_logits)

        safe_pre_logits = safe_pre_logits[:, pad_info[0]: safe_pre_logits.shape[1] - pad_info[1],
                          pad_info[2]: safe_pre_logits.shape[2] - pad_info[3]]
        # you can now do anything with the normal
        safe_pre_logits = torch.nn.functional.interpolate(safe_pre_logits[None, :, :], rgb_origin.shape[:2],
                                                          mode='bilinear').squeeze()
        probs = F.softmax(safe_pre_logits, dim=0)  # 形状 (B, C, H, W)
        pred_labels = torch.argmax(probs, dim=0).cpu().numpy()

        # probs_2 = F.softmax(safe_pre_logits_copy, dim=0)
        # pred_labels_2 = torch.argmax(probs_2, dim=0).cpu()
        # pred_labels_2 = pred_labels_2[pad_info[0]: pred_labels_2.shape[0] - pad_info[1],
        #                   pad_info[2]: pred_labels_2.shape[1] - pad_info[3]].numpy()
        # pred_labels_2 = cv2.resize(pred_labels_2, (rgb_origin.shape[1], rgb_origin.shape[0]), interpolation=cv2.INTER_NEAREST)

        seg_metric.process(torch.tensor(pred_labels), torch.tensor(safe_gt))
        seg_metrics, print_str = seg_metric.compute_metrics(seg_metric.results)
        results_dict[img] = seg_metrics['mIoU']
    return results_dict


def run_eval(model_path, log_txt, cfg_file):
    # MODEL_TYPE['ViT-Small'][
    #     'ckpt_file'] = model_path
    cfg = Config.fromfile(cfg_file)
    if 'refine' in model_path:
        cfg.with_refine = True
    if 'nohidden' in model_path:
        cfg.flow_ablation = [True, True, False]
    if 'withhidden' in model_path:
        cfg.flow_ablation = [True, True, True]
    if 'noall' in model_path:
        cfg.flow_ablation = [False, False, False]
    if '2gru' in model_path:
        cfg.model.decode_head.type = 'RAFTDepthNormalSafe2DPT5'
    model = metric3d_vit_small_(pretrain=True, cfg=cfg, ckpt_file=model_path)
    model.cuda().eval()
    # results_dict = eval_sd_single(model)
    # sorted_dict = dict(sorted(results_dict.items(), key=lambda item: item[1]))
    #
    # # 输出排序后的字典
    # for key, value in sorted_dict.items():
    #     print(f"{key}: {value}")
    seg_metrics1, print_str1 = eval_wilduav(model)
    seg_metrics2, print_str2 = eval_sd(model)
    with open(log_txt, 'a') as f:
        f.write('-' * 50 + '\n')
        f.write(model_path + '\n')
        f.write('Eval on wilduav:' + '\n')
        f.write(print_str1 + '\n')
        f.write(str(seg_metrics1) + '\n')
        f.write('Eval on sd:' + '\n')
        f.write(print_str2 + '\n')
        f.write(str(seg_metrics2) + '\n')
    return (seg_metrics1['mIoU'] + seg_metrics2['mIoU']) / 2
    # return 11


if __name__ == '__main__':
    import cv2
    import numpy as np
    import shutil
    import tqdm
    cfg_file = f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py'
    root_dir = '/home/vector/Tan/xunfeidan/Metric3D/'
    model_paths = [
        # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_0.9-nodice_nohidden_freezedepth_nokl_biglrema_emafreeze/20250221_084910/best_safe_ckpt',
        # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_withhidden_freezedepth_dice/20250226_032549/best_safe_ckpt',
        # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw1-1_reverse_bs8_multi_nosemi_withhidden_freezedepth/20250226_060903/best_safe_ckpt',
        'final_model',
        # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_bs8_multi_nosemi_nohidden_freezedepth_focal_dice/20250228_061138/best_safe_ckpt',
    ]
    log_txt = 'Eval_best_model_fl.txt'
    results_dict = {}
    for p_model_path in tqdm.tqdm(model_paths):
        # dict_key = p_model_path.split('/')[2]
        p_model_path_ = root_dir + p_model_path
        for model_name in os.listdir(p_model_path_):
            # dict_key = p_model_path.split('/')[2] + '_' + model_name
            dict_key = model_name
            mmiou = run_eval(p_model_path + '/' + model_name, log_txt, cfg_file)
            results_dict[dict_key] = mmiou
    sorted_dict = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)
    with open(log_txt, 'a') as log_file:
        for key, value in sorted_dict:
            log_file.write(f'*' * 100 + '\n')
            log_file.write(f'{key}: {value}\n')
