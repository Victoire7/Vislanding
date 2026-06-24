import os
import json
import numpy as np
from typing import List, Dict


def sample_mid_air_dataset(root_path: str, sample_interval: int = 10, test_ratio: float = 0.1):
    """
    采样Mid-Air数据集并生成训练和测试JSON

    Args:
        root_path: 数据集根目录
        sample_interval: 采样间隔帧数
        test_ratio: 测试集比例
    """
    # 文件夹路径
    color_path = os.path.join(root_path, 'color_left')
    depth_path = os.path.join(root_path, 'depth')
    normal_path = os.path.join(root_path, 'normals')
    sem_path = os.path.join(root_path, 'segmentation')

    # 存储训练和测试数据的列表
    train_files = []
    test_files = []

    # 遍历轨迹文件夹
    for trajectory in os.listdir(color_path):
        traj_color_path = os.path.join(color_path, trajectory)
        traj_depth_path = os.path.join(depth_path, trajectory)
        traj_normal_path = os.path.join(normal_path, trajectory)
        traj_sem_path = os.path.join(sem_path, trajectory)

        # 获取所有图像文件
        color_images = sorted([f for f in os.listdir(traj_color_path) if f.endswith('.JPEG')])
        depth_images = sorted([f for f in os.listdir(traj_depth_path) if f.endswith('.PNG')])
        normal_images = sorted([f for f in os.listdir(traj_normal_path) if f.endswith('.PNG')])
        sem_images = sorted([f for f in os.listdir(traj_sem_path) if f.endswith('.PNG')])

        # 计算测试集起始索引
        test_start_idx = int(len(color_images) * (1 - test_ratio))

        # 采样数据
        sampled_indices = list(range(0, len(color_images), sample_interval))

        for idx in sampled_indices:
            # 确定是训练集还是测试集
            is_test = idx >= test_start_idx

            # 获取对应图像文件名
            color_img = color_images[idx]
            depth_img = depth_images[idx]
            normal_img = normal_images[idx]
            sem_img = sem_images[idx]

            # 获取图像尺寸
            from PIL import Image
            img = Image.open(os.path.join(traj_color_path, color_img))
            img_w, img_h = img.size

            # 构建数据条目
            data_entry = {
                'rgb': os.path.join(color_path, trajectory, color_img),
                'depth': os.path.join(depth_path, trajectory, depth_img),
                'normal': os.path.join(normal_path, trajectory, normal_img),
                'sem': os.path.join(sem_path, trajectory, sem_img),
                'cam_in': [img_w / 2, img_h / 2, img_w / 2, img_h / 2]  # [fx, fy, cx, cy]
            }

            # 添加到对应数据集
            if is_test:
                test_files.append(data_entry)
            else:
                train_files.append(data_entry)

    # 写入训练JSON
    with open(os.path.join(root_path, 'MidAir_train.json'), 'w') as f:
        json.dump({'files': train_files}, f, indent=2)

    # 写入测试JSON
    with open(os.path.join(root_path, 'MidAir_val.json'), 'w') as f:
        json.dump({'files': test_files}, f, indent=2)

    print(f"训练集样本数: {len(train_files)}")
    print(f"测试集样本数: {len(test_files)}")


# 使用示例
root_path = '/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/MidAir/PLE_training/spring'
sample_mid_air_dataset(root_path)