import os
import json
import math

import cv2

common_root = '/home/vector/Tan/xunfeidan/Metric3D/training/data/DDOS/data'

# split = ['train', 'test', 'validation']
split = ['test', 'validation']

mid = 'park'

save_root = '/home/vector/Tan/xunfeidan/Metric3D/training/data/DDOS'


def cal_intrinsic(image, FOV=90):
    h, w, _ = image.shape
    # 图像中心点坐标
    c_x = w / 2
    c_y = h / 2

    # 将视场角从度转换为弧度
    FOV_x_radians = math.radians(FOV)

    # 计算x方向的焦距（以像素为单位）
    f_x = w / (2 * math.tan(FOV_x_radians / 2))

    # 计算y方向的焦距（以像素为单位）
    # 由于传感器的宽高比是w:h，我们可以计算y方向上的视场角
    FOV_y_radians = 2 * math.atan((h / 2) / f_x)
    f_y = h / (2 * math.tan(FOV_y_radians / 2))

    # 返回焦距和图像中心点坐标
    return [f_x, f_y, c_x, c_y]


for sp in split:
    cnt = 0
    invalid_cnt = 0
    file_list = []
    root_dir = os.path.join(common_root, sp, mid)
    print(f"Record data ... {sp} DDOS")
    for id_dir in os.listdir(root_dir):
        depth_root = os.path.join(root_dir, id_dir, 'depth')
        image_root = os.path.join(root_dir, id_dir, 'image')
        sn_root = os.path.join(root_dir, id_dir, 'surfacenormals')
        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name)
            depth_path = os.path.join(depth_root, image_name)
            sn_path = os.path.join(sn_root, image_name)

            image = cv2.imread(image_path)
            cam_in = cal_intrinsic(image)
            info_dict = {}
            if os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(sn_path):
                info_dict['rgb'] = image_path
                info_dict['depth'] = depth_path
                info_dict['cam_in'] = cam_in
                info_dict['normal'] = sn_root
                file_list.append(info_dict)

            try:
                assert os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(sn_path)
                cnt += 1
            except:
                invalid_cnt += 1
                continue

    print(cnt, invalid_cnt)
    save_path = os.path.join(save_root, f'ddos_{sp}.json')
    with open(save_path, 'w') as fj:
        json.dump({'files': file_list}, fj)
