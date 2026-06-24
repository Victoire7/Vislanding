dependencies = ['torch', 'torchvision']

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import matplotlib.pyplot as plt

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)


def visualize_depth_comparison(pred_depth, gt_depth):
    """
    可视化预测深度图和真实深度图的对比。

    参数：
    - pred_depth: 预测的深度图（numpy 数组，单位为米）
    - gt_depth: 真实的深度图（numpy 数组，单位为米）
    """

    # 确保深度图是二维的
    if pred_depth.ndim != 2 or gt_depth.ndim != 2:
        raise ValueError("输入的深度图必须是二维数组。")

    # 归一化深度图以便可视化
    pre_min = np.min(pred_depth)
    pre_max = np.max(pred_depth)
    gt_min = np.min(gt_depth)
    gt_max = np.max(gt_depth)
    norm_min = min(pre_min, gt_min)
    norm_max = max(pre_max, gt_max)
    pred_depth_norm = (pred_depth - norm_min) / (norm_max - norm_min)
    gt_depth_norm = (gt_depth - norm_min) / (norm_max - norm_min)

    # 创建子图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 显示预测深度图
    ax[0].imshow(pred_depth_norm, cmap='jet', vmin=0, vmax=1)
    ax[0].set_title('Predicted Depth')
    ax[0].axis('off')  # 不显示坐标轴

    # 显示真实深度图
    ax[1].imshow(gt_depth_norm, cmap='jet', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Depth')
    ax[1].axis('off')  # 不显示坐标轴

    # 显示颜色条
    plt.colorbar(ax[0].imshow(pred_depth_norm, cmap='jet', vmin=0, vmax=1), ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(ax[1].imshow(gt_depth_norm, cmap='jet', vmin=0, vmax=1), ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


MODEL_TYPE = {
    'ConvNeXt-Tiny': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
    },
    'ConvNeXt-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
    },
    'ViT-Small': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
        'ckpt_file': 'weight/metric_depth_vit_small_800k.pth',
    },
    'ViT-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        'ckpt_file': 'weight/metric_depth_vit_large_800k.pth',
    },
    'ViT-giant2': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
    },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    See hubconf.py in this repository for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model


def metric3d_convnext_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    See hubconf.py in this repository for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model


def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    See hubconf.py in this repository for usage examples.
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
        # model.load_state_dict(
        #     torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
        #     strict=False,
        # )
        model.load_state_dict(torch.load(ckpt_file)['model_state_dict'], strict=False)
    return model


def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    See hubconf.py in this repository for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model


def metric3d_vit_giant2(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    See hubconf.py in this repository for usage examples.
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model

def show_normal(normal):
    def show_pixel_value(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 获取当前鼠标位置的像素值
            if 0 <= x < normal.shape[1] and 0 <= y < normal.shape[0]:  # 确保坐标在图像范围内
                pixel_value = normal[y, x]
                # 在图像上添加文本
                text = f"Pixel: {pixel_value}"
                img_with_text = normal.copy()
                cv2.putText(img_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return img_with_text
        return normal

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', show_pixel_value)

    # 显示图像
    while True:
        img_display = show_pixel_value(cv2.EVENT_MOUSEMOVE, 0, 0, None, None)  # 初始化显示
        cv2.imshow('Image', img_display)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import cv2
    import numpy as np

    MODEL_TYPE['ViT-Small'][
        'ckpt_file'] = 'training/work_dirs/vit.raft5.small.wilduav_com/20250106_003614/ckpt/step00002250.pth'
    #### prepare data
    # rgb_file = '/home/vector/Tan/xunfeidan/Depth-Anything-V2/metric_depth/vis_pointcloud/real_test/images/frame_113.jpg'
    # depth_file = 'data/kitti_demo/depth/0000000050.png'
    rgb_file = "training/data/WildUAV/mapping_set/seq03/img/000401.jpg"
    depth_gt_p = "training/data/WildUAV/mapping_set/seq03/depth/000401.npy"
    depth_gt = np.load(depth_gt_p)

    depth_file = None
    intrinsic = [4548.913814319164, 4548.913814319164, 2647.233923204192, 1964.0013181957042]
    # intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    # intrinsic = [600, 600, 480, 270]
    # gt_depth_scale = 256.0
    gt_depth_scale = 1
    rgb_origin_ = cv2.imread(rgb_file)
    H, W, _ = rgb_origin_.shape
    rgb_origin = rgb_origin_[:, :, ::-1]

    model_s = 'small'

    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    # scale = max(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
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

    ###################### canonical camera space ######################
    # inference
    # model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    if model_s == 'small':
        model = metric3d_vit_small(pretrain=True)
    elif model_s == 'large':
        model = metric3d_vit_large(pretrain=True)
    model.cuda().eval()
    # save_dir = "normal_result/" + "test_1228/" + rgb_file.split('/')[-1].split('.')[0] + '/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    # np.save(save_dir + rgb_file.split('/')[-1] + f'_pred_depth_raw_{model_s}.npy', pred_depth.cpu().numpy())

    # un pad
    pred_depth = pred_depth.squeeze()
    a = pad_info[0]
    b = pred_depth.shape[0] - pad_info[1]
    c = pad_info[2]
    d = pred_depth.shape[1] - pad_info[3]
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
    pred_depth = pred_depth / canonical_to_real_scale  # now the depth is metric
    max_pre_d = torch.max(pred_depth)
    min_pre_d = torch.min(pred_depth)
    pred_depth = torch.clamp(pred_depth, 1, 220)

    #### you can now do anything with the metric depth
    # such as evaluate predicted depth

    # np.save(save_dir + rgb_file.split('/')[-1] + f'_pred_depth_{model_s}.npy', pred_depth.cpu().numpy())
    if depth_gt is not None:
        # gt_depth = cv2.imread(depth_file, -1)
        gt_depth = depth_gt / gt_depth_scale
        gt_max = np.max(gt_depth)
        gt_min = np.min(gt_depth)
        gt_depth = torch.from_numpy(gt_depth).float().cuda()

        assert gt_depth.shape == pred_depth.shape

        mask = (gt_depth > 1e-8)
        aa = torch.abs(pred_depth[mask] - gt_depth[mask]).sum()
        bb = torch.abs(pred_depth[mask] - gt_depth[mask]).mean()
        abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
        print('abs_rel_err:', abs_rel_err.item())

        visualize_depth_comparison(pred_depth.cpu().numpy(), gt_depth.cpu().numpy())

    #### normal are also available
    if 'prediction_normal' in output_dict:  # only available for Metric3Dv2, i.e. vit model
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
        # a = pred_normal_vis.astype(np.uint8)
        # cv2.imwrite("normal_result/" + rgb_file.split('/')[-1] + '_normal_vis_raw.png', pred_normal_vis)
        # np.save(save_dir + rgb_file.split('/')[-1] + f'_normal_vis_raw_{model_s}.npy', pred_normal_vis)
        # pred_normal_vis = (pred_normal_vis + 1) / 2

        # cv2.imshow("pred_normal_vis", cv2.resize(pred_normal_vis, (W // 4, H // 4)))
        # cv2.waitKey(0)
        show_normal(cv2.resize(pred_normal_vis, (W // 4, H // 4)))
        # cv2.imwrite(save_dir + rgb_file.split('/')[-1], rgb_origin_)
        # cv2.imwrite(save_dir + rgb_file.split('/')[-1] + f'_normal_vis_{model_s}.png',
        #             (pred_normal_vis * 255).astype(np.uint8))
