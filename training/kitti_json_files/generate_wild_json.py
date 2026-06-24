import os
import json
import math
import numpy as np
import cv2
import tqdm

common_root = '/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/WildUAV'

# split = ['train', 'test', 'validation']
split = ['val', 'train']
save_root = common_root
mode = "labled"
save_dict = {}


if mode == "labled":
    sub_root = os.path.join(common_root, 'mapping_set_4xr')
    semantic_root = os.path.join(sub_root, 'semantic_extension')
    for sp in split:
        files_ = []
        with open(os.path.join(sub_root, f'{sp}.txt'), 'r') as f:
            file_list = f.readlines()
        for target in tqdm.tqdm(file_list):
            info_dict = {}
            target = target.strip('\n')
            seq_name = target.split('/')[0]
            img_name = target.split('/')[1]

            img_path = os.path.join(sub_root, seq_name, 'img', img_name)
            depth_path = os.path.join(sub_root, seq_name, 'depth', img_name.replace('jpg','png'))
            meta_path = os.path.join(sub_root, seq_name, 'metadata', img_name.replace('jpg','json'))

            info_dict['rgb'] = img_path
            info_dict['depth'] = depth_path
            meta_refer = json.load(open(meta_path))
            intrinsic_matrix = np.array(meta_refer['intrinsicMatrix'])
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[2, 0]
            cy = intrinsic_matrix[2, 1]
            paras = [fx, fy, cx, cy]
            info_dict['cam_in'] = paras

            semantic_path = os.path.join(semantic_root, seq_name+'_semantic', 'semantic', img_name.replace('jpg','png'))
            info_dict['sem'] = semantic_path
            normal_path = os.path.join(sub_root, seq_name, 'normals', img_name.replace('jpg','png'))
            info_dict['normal'] = normal_path
            files_.append(info_dict)
        save_path = os.path.join(save_root, f'wild_{mode}_{sp}.json')
        with open(save_path, 'w') as fj:
            json.dump({'files': files_}, fj)
else:
    refer_meta_path = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/WildUAV/mapping_set/seq00/metadata/000000.json"
    meta_refer = json.load(open(refer_meta_path))

    intrinsic_matrix = np.array(meta_refer['intrinsicMatrix'])
    refer_H = meta_refer['height']
    refer_W = meta_refer['width']

    scale_H = 2160 / refer_H
    scale_W = 3840 / refer_W

    fx = intrinsic_matrix[0, 0] * scale_W
    fy = intrinsic_matrix[1, 1] * scale_H
    cx = intrinsic_matrix[2, 0] * scale_W
    cy = intrinsic_matrix[2, 1] * scale_H
    paras = [fx, fy, cx, cy]
    files_ = []
    sub_root = os.path.join(common_root, 'video_set')
    for root, dirs, files in os.walk(sub_root):
        for file in files:
            if file.endswith('.jpg'):
                files_.append({'rgb': os.path.join(root, file), 'cam_in': paras})
    save_dict['files'] = files_
    save_path = os.path.join(save_root, f'wild_{mode}.json')
    with open(save_path, 'w') as fj:
        json.dump(save_dict, fj)