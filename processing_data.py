"""
下载的数据集解压保存到dataset文件夹中，命名为asoct_dataset，通过如下代码可以自动分配训练集、验证集和测试集，用于分割模型训练，
保存在dataset文件夹中，命名为segmentation_dataset。
"""

import os
import json
import shutil
import random
from PIL import Image
import numpy as np
import sys

# 确保labelme库已安装
try:
    import labelme
except ImportError:
    print("错误: 未找到 labelme 库。请使用 'pip install labelme' 命令进行安装。")
    sys.exit(1)


SOURCE_DATASET_DIR = r"dataset/asoct_dataset"
OUTPUT_DATASET_DIR = r"dataset/segmentation_dataset"
DISEASE_FOLDERS = ['Cataract', 'Normal', 'PACG', 'PACG_Cataract']
ANATOMICAL_PARTS = ['anterior_chamber', 'lens', 'nucleus', 'left_iris', 'right_iris']

# 数据集划分比例
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

RANDOM_SEED = 42


def gather_file_pairs(source_dir, disease_folders):
    """
    遍历源数据目录，收集所有 (原始图像路径, JSON文件路径) 对。
    """
    print("--- 步骤 1: 正在收集合所有图像和JSON文件对... ---")
    file_pairs = []
    for disease in disease_folders:
        disease_path = os.path.join(source_dir, disease)
        if not os.path.isdir(disease_path):
            print(f"警告: 找不到疾病文件夹 '{disease_path}'，已跳过。")
            continue

        original_images_path = os.path.join(disease_path, 'Original Images')
        annotated_images_path = os.path.join(disease_path, 'Annotated Images')

        if not os.path.isdir(original_images_path) or not os.path.isdir(annotated_images_path):
            print(f"警告: 在 '{disease_path}' 中未找到 'Original Images' 或 'Annotated Images' 文件夹，已跳过。")
            continue

        for json_filename in os.listdir(annotated_images_path):
            if not json_filename.endswith('.json'):
                continue

            base_name = os.path.splitext(json_filename)[0]
            json_path = os.path.join(annotated_images_path, json_filename)

            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image_path = os.path.join(original_images_path, base_name + ext)
                if os.path.exists(potential_image_path):
                    image_path = potential_image_path
                    break

            if image_path:
                file_pairs.append((image_path, json_path))
            else:
                print(f"警告: 找不到与 '{json_path}' 对应的原始图像，已跳过。")

    print(f"成功收集到 {len(file_pairs)} 个文件对。")
    return file_pairs


def split_dataset(file_pairs, ratios, seed):
    """
    将文件对列表按比例划分为训练集、验证集和测试集。
    """
    print("\n--- 步骤 2: 正在划分数据集... ---")
    random.seed(seed)
    random.shuffle(file_pairs)

    total_count = len(file_pairs)
    train_count = int(total_count * ratios['train'])
    val_count = int(total_count * ratios['val'])

    test_count = total_count - train_count - val_count

    train_files = file_pairs[:train_count]
    val_files = file_pairs[train_count: train_count + val_count]
    test_files = file_pairs[train_count + val_count:]

    print(f"数据集划分完成:")
    print(f"  - 训练集: {len(train_files)} 个样本")
    print(f"  - 验证集: {len(val_files)} 个样本")
    print(f"  - 测试集: {len(test_files)} 个样本")

    return {'train': train_files, 'val': val_files, 'test': test_files}


def create_output_structure(output_dir, parts):
    """
    创建最终的输出目录结构。
    """
    print("\n--- 步骤 3: 正在创建输出目录结构... ---")
    if os.path.exists(output_dir):
        print(f"发现已存在的输出目录 '{output_dir}'，正在删除...")
        shutil.rmtree(output_dir)

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)

        os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)

        for part in parts:
            os.makedirs(os.path.join(split_path, part), exist_ok=True)

    print(f"输出目录 '{output_dir}' 结构创建成功。")


def process_and_save_files(split_data, output_dir, parts):
    """
    处理每个划分的数据，生成掩码并保存到指定位置。
    """
    print("\n--- 步骤 4: 正在处理文件并生成掩码... ---")
    for split_name, file_pairs in split_data.items():
        print(f"\n--- 正在处理 '{split_name}' 集 ---")
        if not file_pairs:
            print(f"'{split_name}' 集为空，跳过。")
            continue

        count = 0
        total = len(file_pairs)
        for image_path, json_path in file_pairs:
            count += 1
            base_name_with_ext = os.path.basename(image_path)
            base_name = os.path.splitext(base_name_with_ext)[0]
            print(f"  ({count}/{total}) 正在处理: {base_name}")

            dest_image_dir = os.path.join(output_dir, split_name, 'images')
            shutil.copy(image_path, os.path.join(dest_image_dir, base_name_with_ext))

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_shape = (data['imageHeight'], data['imageWidth'])

                for part_name in parts:
                    part_shapes = [s for s in data['shapes'] if s['label'] == part_name]

                    if not part_shapes:
                        mask_array = np.zeros(img_shape, dtype=np.uint8)
                    else:
                        temp_label_map = {part_name: 1}
                        lbl = labelme.utils.shapes_to_label(
                            img_shape=img_shape,
                            shapes=part_shapes,
                            label_name_to_value=temp_label_map,
                        )
                        mask_array = (lbl * 255).astype(np.uint8)

                    mask_dir = os.path.join(output_dir, split_name, part_name)
                    mask_path = os.path.join(mask_dir, f"{base_name}.png")
                    Image.fromarray(mask_array).save(mask_path)

            except Exception as e:
                print(f"    错误: 处理文件 '{json_path}' 时发生错误: {e}")
                print(f"    跳过文件对: ('{image_path}', '{json_path}')")
                continue

def main():
    """
    主函数，协调整个处理流程。
    """
    if not os.path.isdir(SOURCE_DATASET_DIR):
        print(f"错误: 源数据集目录 '{SOURCE_DATASET_DIR}' 不存在。请检查路径配置。")
        return

    all_files = gather_file_pairs(SOURCE_DATASET_DIR, DISEASE_FOLDERS)
    if not all_files:
        print("错误: 未找到任何有效的图像-JSON文件对。请检查源数据集结构和内容。")
        return

    split_datasets = split_dataset(all_files, SPLIT_RATIOS, RANDOM_SEED)

    create_output_structure(OUTPUT_DATASET_DIR, ANATOMICAL_PARTS)

    process_and_save_files(split_datasets, OUTPUT_DATASET_DIR, ANATOMICAL_PARTS)

    print("\n--- 所有处理已成功完成！ ---")
    print(f"最终的数据集已保存在: {OUTPUT_DATASET_DIR}")


if __name__ == '__main__':
    main()
