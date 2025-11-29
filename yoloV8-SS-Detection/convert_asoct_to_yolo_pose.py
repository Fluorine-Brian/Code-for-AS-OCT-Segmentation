"""
将LabelMe格式的AS-OCT标注数据转换为YOLOv11 Pose格式
支持4个类别合并训练: Cataract, Normal, PACG, PACG_Cataract

每个样本检测2个关键点:
- left_scleral_spur (左侧巩膜突)
- right_scleral_spur (右侧巩膜突)

所有类别合并为单个类别 'asoct' 进行训练
"""

import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# 配置
CATEGORIES = ['Cataract', 'Normal', 'PACG', 'PACG_Cataract']
SOURCE_BASE = "datasets"
OUTPUT_DIR = "datasets/ASOCT_YOLO"
TRAIN_RATIO = 0.8  # 80% 训练集, 20% 验证集

# 定义类别(所有类别合并为一个)
CLASSES = ["asoct"]  # 只有一个类别

# 关键点名称映射
KEYPOINT_LABELS = [
    "left_scleral_spur",
    "right_scleral_spur"
]


def extract_keypoints_from_json(json_data, img_width, img_height):
    """
    从JSON中提取关键点
    返回: [x1, y1, v1, x2, y2, v2]
    v=2表示可见, v=0表示未标注
    """
    keypoints = []
    kp_dict = {}

    for shape in json_data['shapes']:
        label = shape['label']
        shape_type = shape['shape_type']

        # 提取关键点
        if shape_type == 'point':
            if label in KEYPOINT_LABELS:
                point = shape['points'][0]
                kp_dict[label] = point

    # 按照预定义顺序组织关键点
    for kp_label in KEYPOINT_LABELS:
        if kp_label in kp_dict:
            x, y = kp_dict[kp_label]
            # 归一化到0-1
            x_norm = x / img_width
            y_norm = y / img_height
            # 确保在有效范围内
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            keypoints.extend([x_norm, y_norm, 2])  # 2表示可见
        else:
            keypoints.extend([0, 0, 0])  # 0表示未标注

    return keypoints


def get_bbox_from_polygons(json_data, img_width, img_height):
    """
    从多边形标注中计算包围框
    返回: [x_center, y_center, width, height] (归一化)
    """
    all_points = []

    for shape in json_data['shapes']:
        if shape['shape_type'] == 'polygon':
            for point in shape['points']:
                all_points.append(point)

    if not all_points:
        # 如果没有多边形,使用整个图像作为边界框
        return [0.5, 0.5, 1.0, 1.0]

    all_points = np.array(all_points)
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])

    # 计算中心点和宽高(归一化)
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # 确保在有效范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return [x_center, y_center, width, height]


def convert_labelme_to_yolo(json_path, img_width, img_height):
    """
    将单个LabelMe JSON转换为YOLO格式
    格式: <class> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v>
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 获取图像尺寸
    img_h = json_data.get('imageHeight', img_height)
    img_w = json_data.get('imageWidth', img_width)

    # 获取边界框
    bbox = get_bbox_from_polygons(json_data, img_w, img_h)

    # 获取关键点
    keypoints = extract_keypoints_from_json(json_data, img_w, img_h)

    # 组合为YOLO格式: class_id + bbox + keypoints
    yolo_line = [0] + bbox + keypoints  # class_id=0 (所有类别统一为asoct)

    # 格式化为字符串
    yolo_str = ' '.join([f'{x:.6f}' if isinstance(x, float) else str(x) for x in yolo_line])

    return yolo_str


def collect_all_files():
    """
    收集所有类别的标注文件
    返回: [(json_path, img_path, category), ...]
    """
    all_files = []

    for category in CATEGORIES:
        label_dir = Path(SOURCE_BASE) / category / 'Annotated Images'
        image_dir = Path(SOURCE_BASE) / category / 'Original Images'

        if not label_dir.exists():
            print(f"  警告: 目录不存在 - {label_dir}")
            continue

        json_files = list(label_dir.glob('*.json'))

        for json_file in json_files:
            # 获取对应的图片文件
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            img_name = json_data.get('imagePath', '')
            if not img_name:
                continue

            img_path = image_dir / img_name

            if img_path.exists():
                all_files.append((json_file, img_path, category))
            else:
                print(f"  警告: 图片不存在 - {img_path}")

    return all_files


def main():
    print("=" * 70)
    print("LabelMe to YOLOv11 Pose 数据转换工具 - 多类别合并版")
    print("=" * 70)

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 收集所有文件
    print("\n收集数据文件...")
    all_files = collect_all_files()

    if len(all_files) == 0:
        print("错误: 未找到任何有效的数据文件!")
        return

    # 按类别统计
    category_counts = {}
    for _, _, category in all_files:
        category_counts[category] = category_counts.get(category, 0) + 1

    print(f"\n找到 {len(all_files)} 个有效样本:")
    print("-" * 70)
    for category, count in category_counts.items():
        print(f"  {category:20s}: {count:4d} 张")
    print("-" * 70)

    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        all_files,
        train_size=TRAIN_RATIO,
        random_state=42,
        shuffle=True
    )

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_files)} 张 ({TRAIN_RATIO*100:.0f}%)")
    print(f"  验证集: {len(val_files)} 张 ({(1-TRAIN_RATIO)*100:.0f}%)")

    # 处理数据
    def process_split(files, split_name):
        print(f"\n处理 {split_name} 数据...")
        success_count = 0
        error_count = 0

        for json_file, img_path, category in files:
            try:
                # 读取JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                img_name = json_data['imagePath']

                # 为避免不同类别的同名文件冲突,在文件名前加类别前缀
                prefix = category.lower()
                new_img_name = f"{prefix}_{img_name}"
                label_name = new_img_name.rsplit('.', 1)[0] + '.txt'

                # 转换为YOLO格式
                img_h = json_data.get('imageHeight', 1868)
                img_w = json_data.get('imageWidth', 2135)
                yolo_line = convert_labelme_to_yolo(json_file, img_w, img_h)

                # 保存图片
                output_img_path = output_path / 'images' / split_name / new_img_name
                shutil.copy2(img_path, output_img_path)

                # 保存标注
                output_label_path = output_path / 'labels' / split_name / label_name
                with open(output_label_path, 'w') as f:
                    f.write(yolo_line + '\n')

                success_count += 1

                if success_count % 50 == 0:
                    print(f"  已处理: {success_count}/{len(files)}")

            except Exception as e:
                print(f"  错误: {json_file.name} - {str(e)}")
                error_count += 1

        print(f"  完成: 成功 {success_count}, 失败 {error_count}")
        return success_count, error_count

    # 处理训练集和验证集
    train_success, train_error = process_split(train_files, 'train')
    val_success, val_error = process_split(val_files, 'val')

    # 打印总结
    print("\n" + "=" * 70)
    print("转换完成!")
    print("=" * 70)
    print(f"\n总计: {len(all_files)} 个文件")
    print(f"成功: {train_success + val_success} 个")
    print(f"失败: {train_error + val_error} 个")

    print(f"\n输出目录: {output_path.absolute()}")
    print(f"  训练集: {train_success} 张")
    print(f"  验证集: {val_success} 张")

    print(f"\n数据集信息:")
    print(f"  类别数: 1 (asoct - 合并所有类别)")
    print(f"  关键点数: {len(KEYPOINT_LABELS)}")
    print(f"  关键点: {', '.join(KEYPOINT_LABELS)}")

    print(f"\n原始类别分布:")
    for category, count in category_counts.items():
        percentage = (count / len(all_files)) * 100
        print(f"  {category:20s}: {count:4d} 张 ({percentage:.1f}%)")

    print("\n下一步:")
    print("1. 检查生成的数据: datasets/ASOCT_YOLO/")
    print("2. 使用配置文件: datasets/asoct-pose.yaml")
    print("3. 开始训练: python train_pose.py")


if __name__ == '__main__':
    main()
