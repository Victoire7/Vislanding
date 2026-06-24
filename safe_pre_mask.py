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


if __name__ == '__main__':
    import cv2
    import numpy as np
    import shutil

    MODEL_TYPE['ViT-Small'][
        'ckpt_file'] = '/home/vector/Tan/xunfeidan/Metric3D/training/work_dirs/vit.raft5.small.wilduav_safe_lw2-1_reverse_bs8_multi_nosemi_nohidden_freezedepth/20250222_031223/best_safe_ckpt/student_step00004400_86.04.pth'

    root_dir = '/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/WildUAV/mapping_set_compress'
    seq_info = ['seq00', 'seq01', 'seq02', 'seq03']

    # rgb_file = "/home/vector/Tan/xunfeidan/Metric3D/training/data/WildUAV/mapping_set_4xr/seq03/img/000055.jpg"
    rgb_file = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/Semantic_Drone_Dataset/dataset/rgb_compress/060.jpg"

    # gt_depth_path = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/WildUAV/mapping_set_compress/seq03/depth/000020.png"

    depth_file = None
    # intrinsic = [4545.239894100246 / 4, 4545.239894100246 / 4, 2643.5350965445928 / 4, 1965.6392125747034 / 4]
    intrinsic = [4545.239894100246 / 4, 4545.239894100246 / 4, 750, 500]
    model = metric3d_vit_small(pretrain=True)
    model.cuda().eval()

    rgb, rgb_origin, intrinsic, pad_info = prepare_input(rgb_file, intrinsic)

    ###################### canonical camera space ######################
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})

    safe_pre_logits = output_dict['safe_prediction'][0]
    safe_pre_logits_copy = copy.deepcopy(safe_pre_logits)

    # 原有预处理保持不变
    safe_pre_logits = safe_pre_logits[:, pad_info[0]: safe_pre_logits.shape[1] - pad_info[1],
                      pad_info[2]: safe_pre_logits.shape[2] - pad_info[3]]
    safe_pre_logits = torch.nn.functional.interpolate(safe_pre_logits[None, :, :], rgb_origin.shape[:2],
                                                      mode='bilinear').squeeze()
    probs = F.softmax(safe_pre_logits, dim=0)
    pred_labels = torch.argmax(probs, dim=0).cpu().numpy()

    probs_2 = F.softmax(safe_pre_logits_copy, dim=0)
    pred_labels_2 = torch.argmax(probs_2, dim=0).cpu()
    pred_labels_2 = pred_labels_2[pad_info[0]: pred_labels_2.shape[0] - pad_info[1],
                    pad_info[2]: pred_labels_2.shape[1] - pad_info[3]].numpy()
    pred_labels_2 = cv2.resize(pred_labels_2, (rgb_origin.shape[1], rgb_origin.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # 加载GT并处理
    if "Semantic" in rgb_file:
        safe_gt = rgb_file.replace('rgb_compress', 'safe_ann').replace('jpg', 'png')
        gt = cv2.imread(safe_gt, -1)
        gt = gt == 0
    else:
        safe_gt = rgb_file.replace('img', 'safe_ann').replace('jpg', 'png')
        gt = cv2.imread(safe_gt)
        safe_color = np.array([0, 0, 0])
        safe_mask = np.all(gt == safe_color, axis=-1)
        gt = safe_mask  # 注意这里可能需要根据实际标注调整逻辑

    # 创建可视化画布
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 16))

    # ================== 修改部分开始 ================== #
    # ax4：GT叠加在原图
    rgb_bgr = rgb_origin.copy()  # 保持BGR格式用于OpenCV操作
    gt_overlay = rgb_bgr.copy()
    gt_color = [0, 255, 0]  # BGR格式的绿色
    alpha = 0.5  # 透明度

    # 创建GT蒙版
    gt_mask = gt.astype(bool)
    gt_overlay[gt_mask] = cv2.addWeighted(rgb_bgr, 1, np.full_like(rgb_bgr, gt_color), alpha, 0)[gt_mask]

    ax4.imshow(gt_overlay)
    ax4.set_title('GT Overlay')
    ax4.axis('off')

    # ax6：处理预测结果并叠加
    # 生成预测蒙版
    pred_mask = (pred_labels == 0).astype(np.uint8)

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)

    # 过滤小区域 (阈值设为100像素)
    min_area = 2000
    filtered_mask = np.zeros_like(pred_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered_mask[labels == i] = 1

    # 创建叠加效果
    filter_overlay = rgb_bgr.copy()
    filter_color = [0, 255, 0]  # BGR格式的绿色
    alpha = 0.5

    # 应用过滤后的蒙版
    filter_overlay[filtered_mask.astype(bool)] = cv2.addWeighted(rgb_bgr, 1,
                                                                 np.full_like(rgb_bgr, filter_color),
                                                                 alpha, 0)[filtered_mask.astype(bool)]

    ax6.imshow(filter_overlay)
    ax6.set_title('Filtered Prediction Overlay')
    ax6.axis('off')
    # ================== 修改部分结束 ================== #

    # 其他子图保持不变
    ax1.imshow(pred_labels, cmap=plt.cm.colors.ListedColormap(['green', 'black']), vmin=0, vmax=1)
    ax1.set_title('pred seg')
    ax1.axis('off')

    ax2.imshow(pred_labels_2, cmap=plt.cm.colors.ListedColormap(['green', 'black']), vmin=0, vmax=1)
    ax2.set_title('pred seg 2')
    ax2.axis('off')

    ax3.imshow(rgb_origin)
    ax3.set_title('Raw Image')
    ax3.axis('off')

    # 原有ax5保持不变
    highlight = np.zeros_like(rgb_origin)
    highlight[pred_labels == 0] = [0, 255, 0]  # BGR格式
    overlay_image = cv2.addWeighted(rgb_origin, 1, highlight, 0.5, 0)
    ax5.imshow(overlay_image)
    ax5.set_title('Original Prediction Overlay')
    ax5.axis('off')

    plt.show()
