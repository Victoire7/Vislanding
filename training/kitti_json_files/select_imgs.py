import os


def generate_path_list(root_folder):
    """
    遍历 root 文件夹，生成 path_list，其中每个元素为 '一级目录/二级目录' 格式。
    只保留包含其他格式文件的二级目录。
    """
    path_list = []
    for root, dirs, files in os.walk(root_folder):
        # 只关注一级目录和二级目录
        relative_path = os.path.relpath(root, root_folder)
        parts = relative_path.split(os.sep)

        # 检查是否为一级目录/二级目录
        if len(parts) == 2:  # 一级目录/二级目录
            path_list.append(f"{parts[0]}/{parts[1]}")
    return path_list


def filter_paths_from_txt(txt_file, path_list, output_file, n):
    """
    从 txt 文件中读取路径，检查是否在 path_list 中，保留符合条件的前 n 条。
    """
    filtered_lines = []
    with open(txt_file, "r") as file:
        for line in file:
            # 提取每行的路径部分（假设路径为第一个字段）
            path = line.split()[0]
            if path in path_list:
                filtered_lines.append(line)
                if len(filtered_lines) >= n:  # 达到 n 条后停止
                    break

    # 将结果写入新的 txt 文件
    with open(output_file, "w") as file:
        file.writelines(filtered_lines)


def main():
    root_folder = "/home/vector/Tan/xunfeidan/Metric3D/data/kitti/kitti_data"
    input_txt = "eigen_val.txt"
    output_txt = "eigen_val_0.5k.txt"
    n = 500
    path_list = generate_path_list(root_folder)
    filter_paths_from_txt(input_txt, path_list, output_txt, n)
    print(f"已将符合条件的前 {n} 条路径保存到 {output_txt}")


# 执行主函数
if __name__ == "__main__":
    main()
