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
    # rgb_origin = cv2.cvtColor(rgb_origin_, cv2.COLOR_BGR2RGB)
    rgb_origin = rgb_origin_[:, :, ::-1]
    H, W, _ = rgb_origin.shape
    # rgb_origin = rgb_origin_[:, :, ::-1]

    input_size = (616, 1064)  # for vit model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

    # padding = [123.675, 116.28, 103.53]
    padding = [0.0, 0.0, 0.0]
    h, w = rgb.shape[:2]
    # print(f"Original size: {(H, W)}, Resized size: {(h, w)}, Input size: {input_size}, Scale: {scale:.4f}")
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

def post_process_safe_mask(pred_labels, min_area_ratio=0.002):
    """
    对预测结果进行后处理。
    pred_labels: np.array, 形状 (H, W), 值 0 (safe), 1 (unsafe)
    min_area_ratio: 最小面积比例，小于此比例的孤立区域将被视为噪声
    """
    # 转换为 uint8 格式供 OpenCV 使用
    mask = pred_labels.astype(np.uint8)
    
    # --- 步骤 1: 形态学闭运算 ---
    # 填充安全区内部的小孔洞，连接断开的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 步骤 2: 连通域面积过滤 ---
    # 我们要处理的是 safe 区 (值为 0)，所以先取反，让连通域分析去找 safe 的块
    safe_binary = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_binary, connectivity=8)
    
    # 计算最小面积阈值
    h, w = mask.shape
    min_area = h * w * min_area_ratio
    
    refined_mask = np.ones_like(mask) # 默认全是 1 (unsafe)
    
    for i in range(1, num_labels): # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # 只有面积够大的块才保留为 safe (0)
            refined_mask[labels == i] = 0
            
    return refined_mask

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


def eval_wilduav(model, subset):
    seg_metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'],
                           class_names=['safe', 'unsafe'],
                           ignore_index=2)

    # dataset = json.load(open("/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/wild_labled_val_4xr_safe.json"))
    intrinsic = [4545.239894100246 / 4, 4545.239894100246 / 4, 2643.5350965445928 / 4, 1965.6392125747034 / 4]
    root_dir = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/WildUAV/robust_test"
    img_dir = os.path.join(root_dir, subset)
    ann_dir = os.path.join(root_dir, "ann")
    for img in tqdm.tqdm(os.listdir(img_dir)):
        rgb_file = os.path.join(img_dir, img)
        rgb, rgb_origin, intrinsic, pad_info = prepare_input(rgb_file, intrinsic)
        safe_gt_path = os.path.join(ann_dir, img.replace('.jpg', '.png'))
        safe_gt = load_safe_gt(safe_gt_path)

        ###################### canonical camera space ######################
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})

        safe_pre_logits = output_dict['safe_prediction'][-1]

        safe_pre_logits_copy = copy.deepcopy(safe_pre_logits)

        safe_pre_logits = safe_pre_logits[:, pad_info[0]: safe_pre_logits.shape[1] - pad_info[1],
                          pad_info[2]: safe_pre_logits.shape[2] - pad_info[3]]
        # you can now do anything with the normal
        safe_pre_logits = torch.nn.functional.interpolate(safe_pre_logits[None, :, :], rgb_origin.shape[:2],
                                                          mode='bilinear').squeeze()
        probs = F.softmax(safe_pre_logits, dim=0)  # 形状 (B, C, H, W)
        pred_labels = torch.argmax(probs, dim=0).cpu().numpy()
        # pred_labels = post_process_safe_mask(pred_labels, min_area_ratio=0.05)

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


def eval_sd(model, subset):
    seg_metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'],
                           class_names=['safe', 'unsafe'],
                           ignore_index=2)

    # dataset = json.load(open("/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/wild_labled_val_4xr_safe.json"))
    intrinsic = [4545.239894100246 / 6, 4545.239894100246 / 6, 750, 500]
    root_dir = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/Semantic_Drone_Dataset/dataset/robust_test/"
    img_dir = os.path.join(root_dir, subset)
    ann_dir = os.path.join(root_dir, "ann")
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
        # pred_labels = post_process_safe_mask(pred_labels, min_area_ratio=0.05)

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


def run_eval(model_path, log_txt, cfg_file, subset):
    cfg = Config.fromfile(cfg_file)
    if '-depth' in model_path:
        cfg.flow_ablation[0] = False  # depth, normal, fuse_feature
    if '-normal' in model_path:
        cfg.flow_ablation[1] = False  # depth, normal, fuse_feature
    if 'refine' in model_path:
        cfg.with_refine = True
    # if 'nohidden' in model_path:
    #     cfg.flow_ablation = [True, True, False]
    # if 'withhidden' in model_path:
    #     cfg.flow_ablation = [True, True, True]
    # if 'noall' in model_path:
    #     cfg.flow_ablation = [False, False, False]
    if '2gru' in model_path:
        cfg.model.decode_head.type = 'RAFTDepthNormalSafe2DPT5'
    print(f"Evaluating model: {model_path} on subset: {subset}")
    print(cfg.flow_ablation)
    
    model = metric3d_vit_small_(pretrain=True, cfg=cfg, ckpt_file=model_path)
    model.cuda().eval()

    seg_metrics1, print_str1 = eval_wilduav(model, subset)
    seg_metrics2, print_str2 = eval_sd(model, subset)
    
    with open(log_txt, 'a') as f:
        f.write('-' * 50 + '\n')
        f.write(model_path + '\n')
        f.write(f'Eval on wilduav {subset}:' + '\n')
        f.write(print_str1 + '\n')
        f.write(str(seg_metrics1) + '\n')
        f.write(f'Eval on sd {subset}:' + '\n')
        f.write(print_str2 + '\n')
        f.write(str(seg_metrics2) + '\n')
    
    # 返回均值和详细指标字典
    avg_miou = (seg_metrics1['mIoU'] + seg_metrics2['mIoU']) / 2
    return avg_miou, seg_metrics1, seg_metrics2

def print_and_save_robustness_table(all_results, log_txt):
    """
    将所有结果汇总制表输出
    """
    lines = []
    lines.append("\n" + "="*100)
    lines.append(f"{'Robustness Evaluation Summary Table':^100}")
    lines.append("="*100)
    
    # 表头
    header = f"| {'Subset':<20} | {'WU mIoU':<10} | {'WU mPrec':<10} | {'SD mIoU':<10} | {'SD mPrec':<10} | {'Avg mIoU':<10} |"
    lines.append(header)
    lines.append("|" + "-"*22 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")

    for subset, data in all_results.items():
        m1 = data['wu_metrics']
        m2 = data['sd_metrics']
        avg = data['avg_miou']
        
        row = (f"| {subset:<20} | "
               f"{m1['mIoU']:<10.2f} | {m1.get('mPrecision', 0.0):<10.2f} | "
               f"{m2['mIoU']:<10.2f} | {m2.get('mPrecision', 0.0):<10.2f} | "
               f"{avg:<10.2f} |")
        lines.append(row)
    
    table_str = "\n".join(lines)
    print(table_str)
    with open(log_txt, 'a') as f:
        f.write(table_str + "\n")

if __name__ == '__main__':
    import cv2
    import numpy as np
    import shutil
    import tqdm
    import copy # 确保导入了copy
    
    cfg_file = f'/home/vector/Tan/xunfeidan/Metric3D/mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py'
    root_dir = '/home/vector/Tan/xunfeidan/Metric3D/'
    eval_subset = ['Elastic_Transform', 'foggy', 'Gaussian_Noise', 'Impulse_Noise', 'JPEG_Compression', 'Mild_Zoom_Blur', 'Motion_Blur', 'rain', 'raw', 'snow']
    # eval_subset = ['raw']
    model_paths = [
    # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth_rawmetric/20260319_162536/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-depth/20260320_133224/best_safe_ckpt',
    'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-normal/20260321_201422/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth-normal-depth/20260321_052931/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.sd_safe_base-normal/20260321_025833/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.sd_safe_base-depth/20260319_215025/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.sd_safe_base-depth-normal/20260321_002301/best_safe_ckpt',
    # 'training/work_dirs/vit.raft5.small.sd_safe_base/20260318_144817/best_safe_ckpt',
    # 'final_model'
    ]
    
    log_txt = 'Eval_ablation_new.txt'
    
    # 用于排序的字典
    sort_helper_dict = {}
    # 用于制表展示的详细字典
    table_results = {}

    for p_model_path in tqdm.tqdm(model_paths):
        p_model_path_ = root_dir + p_model_path
        for model_name in os.listdir(p_model_path_):
            for subset in eval_subset:
                dict_key = f"{model_name}_{subset}"
                print(f'Evaluating {dict_key}...')
                
                # 执行评测
                full_path = os.path.join(p_model_path, model_name)
                avg_miou, metrics_wu, metrics_sd = run_eval(full_path, log_txt, cfg_file, subset)
                
                # 存储数据
                sort_helper_dict[dict_key] = avg_miou
                table_results[subset] = {
                    'wu_metrics': metrics_wu,
                    'sd_metrics': metrics_sd,
                    'avg_miou': avg_miou
                }
    
    # 1. 输出排序后的结果
    sorted_dict = sorted(sort_helper_dict.items(), key=lambda item: item[1], reverse=True)
    with open(log_txt, 'a') as log_file:
        log_file.write('\n' + '#' * 100 + '\n')
        log_file.write(f"{'Ranked Results (by Avg mIoU)':^100}\n")
        for key, value in sorted_dict:
            log_file.write(f'{key}: {value:.4f}\n')
            
    # 2. 调用制表函数
    print_and_save_robustness_table(table_results, log_txt)
