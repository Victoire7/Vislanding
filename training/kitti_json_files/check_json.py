import os
import json


def check_dataset_paths(root_path):
    """
    检查JSON中的所有路径是否存在

    Args:
        root_path: 数据集根目录
    """
    # 检查JSON文件
    json_files = ['MidAir_train.json', 'MidAir_val.json']

    for json_filename in json_files:
        json_path = os.path.join(root_path, json_filename)

        with open(json_path, 'r') as f:
            data = json.load(f)

        files = data.get('files', [])
        missing_paths = []

        for file_info in files:
            # 检查每个路径
            for key, subfolder in [
                ('rgb', 'color_left'),
                ('depth', 'depth'),
                ('normal', 'normals')
            ]:
                relative_path = file_info.get(key, '')
                full_path = os.path.join(root_path, subfolder, relative_path)

                if not os.path.exists(full_path):
                    missing_paths.append(full_path)

        if missing_paths:
            print(f"错误：以下路径在 {json_filename} 中不存在：")
            for path in missing_paths[:10]:  # 只打印前10个
                print(path)
            if len(missing_paths) > 10:
                print(f"... 共 {len(missing_paths)} 个缺失路径")
            return False

    print("数据集路径检查通过！")
    return True


# 使用示例
if __name__ == "__main__":
    root_path = "/home/vector/Tan/xunfeidan/Metric3D/training/data/MDE/MidAir/PLE_training/spring"
    check_dataset_paths(root_path)