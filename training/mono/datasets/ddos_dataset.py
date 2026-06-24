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


class DDOSDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(DDOSDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # if data_path['sem_path'] is not None:
        #     print(self.data_name)

        curr_rgb, curr_depth, curr_normal, curr_sem, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_sem'], data_batch['curr_cam_model']
        #curr_stereo_depth = data_batch['curr_stereo_depth']
        
        # A patch for stereo depth dataloader (no need to modify specific datasets)
        if 'curr_stereo_depth' in data_batch.keys():
            curr_stereo_depth = data_batch['curr_stereo_depth']
        else:
            curr_stereo_depth = self.load_stereo_depth_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1]) 

        curr_intrinsic = meta_data['cam_in']
        # data augmentation
        transform_paras = dict(random_crop_size = self.random_crop_size) # dict() 
        assert curr_rgb.shape[:2] == curr_depth.shape == curr_normal.shape[:2] == curr_sem.shape
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb, ], 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[curr_intrinsic,], 
                                                                   cam_models=[curr_cam_model, ],
                                                                   normals = [curr_normal, ],
                                                                   other_labels=[curr_sem, curr_stereo_depth],
                                                                   transform_paras=transform_paras)
        # process sky masks
        sem_mask = other_labels[0].int()
        # clip depth map 
        depth_out = self.normalize_depth(depths[0])
        # set the depth of sky region to the invalid
        depth_out[sem_mask==142] = -1 # self.depth_normalize[1] - 1e-6
        # get inverse depth
        inv_depth = self.depth2invdepth(depth_out, sem_mask==142)
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]

        # stereo_depth
        if 'label_scale_factor' not in transform_paras.keys():
            transform_paras['label_scale_factor'] = 1
        stereo_depth_pre_trans = other_labels[1] * (other_labels[1] > 0.3) * (other_labels[1] < 200)
        stereo_depth = stereo_depth_pre_trans * transform_paras['label_scale_factor']
        stereo_depth = self.normalize_depth(stereo_depth)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]        
        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=torch.tensor(pad),
                    data_type=[self.data_type, ],
                    sem_mask=sem_mask.int(),
                    stereo_depth= stereo_depth,
                    normal=normals[0],
                    inv_depth=inv_depth,
                    scale=transform_paras['label_scale_factor'])
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # load data
        curr_rgb, curr_depth, curr_normal, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_cam_model']
        ori_curr_intrinsic = meta_data['cam_in']

        # get crop size
        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb,],  #+ tmpl_rgbs, 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic, ], # * (len(tmpl_rgbs) + 1), 
                                                                   cam_models=[curr_cam_model, ],
                                                                   transform_paras=transform_paras)
        # depth in original size and orignial metric***
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1] # self.clip_depth(depths[0]) #
        inv_depth = self.depth2invdepth(depth_out, np.zeros_like(depth_out, dtype=np.bool))
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        ori_curr_intrinsic_mat = self.intrinsics_list2mat(ori_curr_intrinsic)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]    
        raw_rgb = torch.from_numpy(curr_rgb)
        curr_normal = torch.from_numpy(curr_normal.transpose((2,0,1)))
        

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
                    normal=curr_normal,
                    )
        return data
    
    def process_depth(self, depth, rgb):
        new_depth = np.zeros_like(depth)
        new_depth = depth
        new_depth[new_depth>65534] = 0
        new_depth /= self.metric_scale
        return new_depth
    



# if __name__ == '__main__':
#     from mmcv.utils import Config 
#     cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
#     dataset_i = ApolloscapeDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
#     print(dataset_i)
    