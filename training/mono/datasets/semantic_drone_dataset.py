import os
import torch
import numpy as np
import cv2
from .__base_dataset__ import BaseDataset

class SemanticDroneDataset(BaseDataset):
    """
    专为只包含 RGB 和 Safe Annotation 的自定义数据集设计。
    所有缺失的 3D/其他标注 (Depth, Normal, Stereo 等) 将被自动填充为占位符，以兼容底层管道。
    """
    def __init__(self, cfg, phase, **kwargs):
        super(SemanticDroneDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale

    def load_data_path(self, meta_data):
        """
        重写路径加载逻辑。因为新数据集只有 rgb 和 safe_ann 字段。
        其他路径全部设为 None，避免 os.path.join 报错。
        """
        curr_rgb_path = meta_data['rgb']
        curr_safe_path = meta_data['safe_ann']

        data_path = dict(
            rgb_path=curr_rgb_path,
            safe_path=curr_safe_path,
            depth_path=None,
            sem_path=None,
            normal_path=None,
            disp_path=None,
            depth_mask_path=None,
        )
        return data_path

    def load_batch(self, meta_data, data_path):
        """
        重写 Batch 加载逻辑。遇到 None 路径时生成占位符。
        """
        curr_intrinsic = meta_data['cam_in']
        
        # 1. 加载真实拥有的数据
        curr_rgb = self.load_data(data_path['rgb_path'], is_rgb_img=True)
        if curr_rgb is None:
            self.logger.info(f">>>> RGB 读取失败: {data_path['rgb_path']}")
            
        curr_safe = self.load_safe_label(data_path['safe_path'])
        
        # 获取图像尺寸用于生成占位符
        H, W = curr_rgb.shape[:2]

        # 2. 生成占位符数据 (Dummy Data)
        # 深度图占位: 全为 -1 (表示无效深度)
        curr_depth = np.ones((H, W), dtype=np.float32) * -1.0
        
        # 法向图占位: 全为 0
        curr_normal = np.zeros((H, W, 3), dtype=np.float32)
        
        # 语义分割占位 (假设原本用作过滤天空等): 全为 -1
        curr_sem = np.ones((H, W), dtype=int) * -1
        
        # 双目视差占位: 全为 -1
        curr_stereo_depth = np.ones((H, W), dtype=np.float32) * -1.0

        # 3. 创建相机模型
        curr_cam_model = self.create_cam_model(H, W, curr_intrinsic)       

        data_batch = dict(
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_sem=curr_sem,
            curr_normal=curr_normal,
            curr_safe=curr_safe,
            curr_cam_model=curr_cam_model,
            curr_stereo_depth=curr_stereo_depth,
        )
        return data_batch

    def load_safe_label(self, safe_path):
        """
        加载安全区标注。这部分保持你原来的逻辑。
        """
        if safe_path is None or not os.path.exists(safe_path):
            # 容错：如果找不到掩码，返回全认为是 unsafe (1) 的图
            return np.ones((1, 1), dtype=np.uint8) # 尺寸会在 transform 报错前被处理
            
        image = cv2.imread(safe_path)
        if image is None:
            raise RuntimeError(f"无法读取掩码文件: {safe_path}")

        # 根据你的规则，黑色 [0, 0, 0] 为 Safe，其他为 Unsafe
        safe_color = np.array([0, 0, 0])
        binary_gt = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) # 默认 1 (Unsafe)
        
        safe_mask = np.all(image == safe_color, axis=-1)
        binary_gt[safe_mask] = 0  # 0 为 Safe

        return binary_gt

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)

        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)

        curr_rgb = data_batch['curr_rgb']
        curr_depth = data_batch['curr_depth']
        curr_normal = data_batch['curr_normal']
        curr_sem = data_batch['curr_sem']
        curr_cam_model = data_batch['curr_cam_model']
        curr_safe = data_batch['curr_safe']
        curr_stereo_depth = data_batch['curr_stereo_depth']

        curr_intrinsic = meta_data['cam_in']
        
        # 数据增强与转换
        transform_paras = dict(random_crop_size=self.random_crop_size) 
        
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],
            labels=[curr_depth, ],
            intrinsics=[curr_intrinsic, ],
            cam_models=[curr_cam_model, ],
            normals=[curr_normal, ],
            other_labels=[curr_sem, curr_stereo_depth, curr_safe], # 注意传入顺序
            transform_paras=transform_paras)

        sem_mask = other_labels[0].int()
        depth_out = depths[0] # 我们不需要 normalize_depth，因为它全是 -1
        
        # 逆深度占位
        inv_depth = torch.zeros_like(depth_out)
        
        filename = os.path.basename(meta_data['rgb'])
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :],
                                            size=(cam_models[0].shape[1] // i, cam_models[0].shape[2] // i),
                                            mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]

        stereo_depth = other_labels[1]

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        
        # 安全区标签处理
        safe_target = other_labels[2]
        safe_target[(safe_target != 0) & (safe_target != 1)] = 2 # ignore index
        
        data = dict(input=rgbs[0],
                    weak_input=torch.zeros_like(rgbs[0]),
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=torch.tensor(pad),
                    data_type=[self.data_type, ],
                    sem_mask=sem_mask,
                    stereo_depth=stereo_depth,
                    safe_target=safe_target,
                    normal=normals[0],
                    inv_depth=inv_depth,
                    scale=transform_paras.get('label_scale_factor', 1.0))
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        
        curr_rgb = data_batch['curr_rgb']
        curr_depth = data_batch['curr_depth']
        curr_normal = data_batch['curr_normal']
        curr_cam_model = data_batch['curr_cam_model']
        curr_safe = data_batch['curr_safe'] # 测试时可能用到真实 GT 算指标
        
        ori_curr_intrinsic = meta_data['cam_in']

        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],  
            labels=[curr_depth, ], 
            intrinsics=[ori_curr_intrinsic, ], 
            cam_models=[curr_cam_model, ],
            normals=[curr_normal, ],
            other_labels=[curr_safe, ], # 测试时透传 safe 以备评估
            transform_paras=transform_paras)
            
        depth_out = depths[0]
        inv_depth = torch.zeros_like(depth_out)
        filename = os.path.basename(meta_data['rgb'])

        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        scale_ratio = transform_paras.get('label_scale_factor', 1.0)
        
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :],
                                            size=(cam_models[0].shape[1] // i, cam_models[0].shape[2] // i),
                                            mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]
        
        raw_rgb = torch.from_numpy(curr_rgb)

        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=pad,
                    scale=scale_ratio,
                    raw_rgb=raw_rgb,
                    sample_id=idx,
                    data_path=meta_data['rgb'],
                    inv_depth=inv_depth,
                    normal=normals[0],
                    safe_target=other_labels[0] # 保存测试 GT
                    )
        return data