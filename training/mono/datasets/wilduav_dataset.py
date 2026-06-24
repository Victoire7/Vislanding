import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from .__base_dataset__ import BaseDataset
import h5py


class WildUAVDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(WildUAVDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        # hdf5_root = self.db_info['hdf5_path'] if 'hdf5_path' in self.db_info else None
        # self.hdf5_root = os.path.join(self.db_info['db_root'], self.db_info['data_root'],
        #                               hdf5_root) if hdf5_root is not None else None
        # self.h5_file = h5py.File(self.hdf5_root, 'r')
        # self.train_sample_ids = [s.decode() for s in self.h5_file['splits']['train'][:]]
        # self.test_sample_ids = [s.decode() for s in self.h5_file['splits']['val'][:]]

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)

        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # sample_id = self.train_sample_ids[idx]
        # data_batch = self.load_batch_hdf5(sample_id)
        # if data_path['sem_path'] is not None:
        #     print(self.data_name)

        curr_rgb, curr_depth, curr_normal, curr_sem, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], \
        data_batch['curr_normal'], data_batch['curr_sem'], data_batch['curr_cam_model']
        # curr_stereo_depth = data_batch['curr_stereo_depth']

        # A patch for stereo depth dataloader (no need to modify specific datasets)
        if 'curr_stereo_depth' in data_batch.keys():
            curr_stereo_depth = data_batch['curr_stereo_depth']
        else:
            curr_stereo_depth = self.load_stereo_depth_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1])

        curr_intrinsic = meta_data['cam_in']
        # curr_intrinsic = data_batch['curr_intrinsic']
        # data augmentation
        transform_paras = dict(random_crop_size=self.random_crop_size)  # dict()
        assert curr_rgb.shape[:2] == curr_depth.shape == curr_normal.shape[:2] == curr_sem.shape
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],
            labels=[curr_depth, ],
            intrinsics=[curr_intrinsic, ],
            cam_models=[curr_cam_model, ],
            normals=[curr_normal, ],
            other_labels=[curr_sem, curr_stereo_depth],
            transform_paras=transform_paras)
        # process sky masks
        sem_mask = other_labels[0].int()
        # clip depth map 
        depth_out = self.normalize_depth(depths[0])
        # set the depth of sky region to the invalid
        depth_out[sem_mask == 179] = -1  # self.depth_normalize[1] - 1e-6
        # get inverse depth
        inv_depth = self.depth2invdepth(depth_out, sem_mask == 179)
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        # filename = sample_id.split('/')[-1]
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :],
                                            size=(cam_models[0].shape[1] // i, cam_models[0].shape[2] // i),
                                            mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]

        # stereo_depth
        if 'label_scale_factor' not in transform_paras.keys():
            transform_paras['label_scale_factor'] = 1
        stereo_depth_pre_trans = other_labels[1] * (other_labels[1] > 0.3) * (other_labels[1] < 200)
        stereo_depth = stereo_depth_pre_trans * transform_paras['label_scale_factor']
        stereo_depth = self.normalize_depth(stereo_depth)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=torch.tensor(pad),
                    data_type=[self.data_type, ],
                    sem_mask=sem_mask.int(),
                    stereo_depth=stereo_depth,
                    normal=normals[0],
                    inv_depth=inv_depth,
                    scale=transform_paras['label_scale_factor'])
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # sample_id = self.test_sample_ids[idx]
        # data_batch = self.load_batch_hdf5(sample_id)
        # load data
        curr_rgb, curr_depth, curr_normal, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], \
        data_batch['curr_normal'], data_batch['curr_cam_model']
        ori_curr_intrinsic = meta_data['cam_in']
        # ori_curr_intrinsic = data_batch['curr_intrinsic']

        # get crop size
        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],  # + tmpl_rgbs,
            labels=[curr_depth, ],
            intrinsics=[ori_curr_intrinsic, ],  # * (len(tmpl_rgbs) + 1),
            cam_models=[curr_cam_model, ],
            transform_paras=transform_paras)
        # depth in original size and orignial metric***
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1]  # self.clip_depth(depths[0]) #
        inv_depth = self.depth2invdepth(depth_out, np.zeros_like(depth_out, dtype=np.bool))
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        # filename = sample_id.split('/')[-1]

        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        ori_curr_intrinsic_mat = self.intrinsics_list2mat(ori_curr_intrinsic)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :],
                                            size=(cam_models[0].shape[1] // i, cam_models[0].shape[2] // i),
                                            mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]
        raw_rgb = torch.from_numpy(curr_rgb)
        curr_normal = torch.from_numpy(curr_normal.transpose((2, 0, 1)))

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
                    # data_path=None,
                    inv_depth=inv_depth,
                    normal=curr_normal,
                    )
        return data

    def load_batch_hdf5(self, sample_id):
        curr_rgb = self.h5_file['rgb'][sample_id][:]
        curr_depth = self.h5_file['depth'][sample_id][:]
        curr_sem = self.h5_file['semantic'][sample_id][:]
        meta_str = self.h5_file['meta'][sample_id][()]
        camera_params = json.loads(meta_str)
        curr_intrinsic = list(camera_params.values())
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)
        # get normal labels
        curr_normal = self.load_norm_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1])
        # get depth mask
        depth_mask = curr_depth > 0
        curr_depth[~depth_mask] = -1
        curr_stereo_depth = self.load_stereo_depth_label(None, H=curr_rgb.shape[0],
                                                         W=curr_rgb.shape[1])

        data_batch = dict(
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_sem=curr_sem,
            curr_normal=curr_normal,
            curr_cam_model=curr_cam_model,
            curr_stereo_depth=curr_stereo_depth,
            curr_intrinsic=curr_intrinsic
        )
        return data_batch

    def load_batch(self, meta_data, data_path):
        curr_intrinsic = meta_data['cam_in']
        # load rgb/depth
        curr_rgb, curr_depth = self.load_rgb_depth(data_path['rgb_path'], data_path['depth_path'])
        # get semantic labels
        curr_sem = self.load_sem_label(meta_data['sem'], curr_depth)
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)
        # get normal labels
        curr_normal = self.load_norm_label(data_path['normal_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1])
        # curr_normal = self.load_norm_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1])

        # get depth mask
        # depth_mask = self.load_depth_valid_mask(data_path['depth_mask_path'])
        depth_mask = curr_depth > 0
        curr_depth[~depth_mask] = -1
        # get stereo depth
        curr_stereo_depth = self.load_stereo_depth_label(data_path['disp_path'], H=curr_rgb.shape[0],
                                                         W=curr_rgb.shape[1])

        data_batch = dict(
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_sem=curr_sem,
            curr_normal=curr_normal,
            curr_cam_model=curr_cam_model,
            curr_stereo_depth=curr_stereo_depth,
        )
        return data_batch

    def load_sem_label(self, sem_path, depth=None, sky_id=179) -> np.array:
        H, W = depth.shape
        # if sem_path is not None:
        #     print(self.data_name)
        sem_label = cv2.imread(sem_path, 0) if sem_path is not None \
            else np.ones((H, W), dtype=int) * -1
        if sem_label is None:
            sem_label = np.ones((H, W), dtype=int) * -1
        # set dtype to int before
        sem_label = sem_label.astype(int)
        sem_label[sem_label == 255] = -1

        # mask invalid sky region
        mask_depth_valid = depth > 1e-8
        invalid_sky_region = (sem_label == 179) & (mask_depth_valid)
        if self.data_type in ['lidar', 'sfm', 'denselidar', 'denselidar_nometric']:
            sem_label[invalid_sky_region] = -1
        unique_values = np.unique(sem_label)
        return sem_label

    def load_rgb_depth(self, rgb_path: str, depth_path: str):
        """
        Load the rgb and depth map with the paths.
        """
        rgb = self.load_data(rgb_path, is_rgb_img=True)
        if rgb is None:
            self.logger.info(f'>>>>{rgb_path} has errors.')

        depth = self.load_data(depth_path)
        if depth is None:
            self.logger.info(f'{depth_path} has errors.')

        # self.check_data(dict(
        #     rgb_path=rgb,
        #     depth_path=depth,
        # ))
        depth = depth.astype(np.float32) / 65535.0 * 220.0
        # if depth.shape != rgb.shape[:2]:
        #     print(f'no-equal in {self.data_name}')
        #     depth = cv2.resize(depth, rgb.shape[::-1][1:])

        depth = self.process_depth(depth, rgb)
        return rgb, depth

    def load_norm_label(self, norm_path, H, W):
        if norm_path is None:
            norm_gt = np.zeros((H, W, 3)).astype(np.float32)
        else:
            norm_gt = cv2.imread(norm_path)

            norm_gt = np.array(norm_gt).astype(np.uint8)
            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm_gt = norm_gt * norm_valid_mask

        return norm_gt

    def load_data_path(self, meta_data):
        curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
        curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
        curr_norm_path = os.path.join(self.data_root, meta_data['normal'])
        curr_sem_path = os.path.join(self.data_root, meta_data['sem'])
        # curr_sem_path = os.path.join(self.sem_root, meta_data['sem']) \
        #     if self.sem_root is not None and ('sem' in meta_data) and (meta_data['sem'] is not None)  \
        #     else None

        curr_depth_mask_path = os.path.join(self.depth_mask_root, meta_data['depth_mask']) \
            if self.depth_mask_root is not None and ('depth_mask' in meta_data) and (meta_data['depth_mask'] is not None)  \
            else None

        if ('disp' in meta_data) and (meta_data['disp'] is not None) and (self.disp_root is not None):
            if isinstance(meta_data['disp'], dict):
                curr_disp_path = {}
                for k,v in meta_data['disp'].items():
                    curr_disp_path[k] = os.path.join(self.disp_root, v)
            else:
                curr_disp_path = os.path.join(self.disp_root, meta_data['disp'])
        else:
            curr_disp_path = None

        data_path=dict(
            rgb_path=curr_rgb_path,
            depth_path=curr_depth_path,
            sem_path=curr_sem_path,
            normal_path=curr_norm_path,
            disp_path=curr_disp_path,
            depth_mask_path=curr_depth_mask_path,
            )
        return data_path
