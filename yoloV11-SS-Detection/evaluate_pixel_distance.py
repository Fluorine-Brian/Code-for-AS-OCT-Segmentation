"""
AS-OCT巩膜突关键点检测评估脚本
计算预测关键点与真实标注之间的像素距离

评估指标:
1. 平均像素距离 (Mean Pixel Distance, MPD)
2. 标准差 (Standard Deviation, STD)
3. 中位数距离 (Median Distance)
4. PCK@Xpx (Percentage of Correct Keypoints at X pixels)
"""

import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt


def load_ground_truth_from_txt(label_path, img_width, img_height):
    """
    从YOLO格式的txt文件加载真实关键点
    返回: [(x1, y1), (x2, y2)] 像素坐标
    """
    if not label_path.exists():
        return None

    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = list(map(float, line.split()))

    # YOLO格式: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
    if len(parts) < 11:
        return None

    # 提取关键点 (归一化坐标)
    kp1_x_norm, kp1_y_norm, kp1_v = parts[5], parts[6], parts[7]
    kp2_x_norm, kp2_y_norm, kp2_v = parts[8], parts[9], parts[10]

    keypoints = []

    # left_scleral_spur
    if kp1_v > 0:  # 可见
        x = kp1_x_norm * img_width
        y = kp1_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    # right_scleral_spur
    if kp2_v > 0:  # 可见
        x = kp2_x_norm * img_width
        y = kp2_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    return keypoints


def calculate_pixel_distance(pred_kp, gt_kp):
    """
    计算两个关键点之间的欧氏距离（像素）
    pred_kp: (x, y) 预测坐标
    gt_kp: (x, y) 真实坐标
    返回: 距离（像素）
    """
    if pred_kp is None or gt_kp is None:
        return None

    pred_x, pred_y = pred_kp
    gt_x, gt_y = gt_kp

    distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    return distance


def evaluate_model(model_path, val_images_dir, val_labels_dir, conf_threshold=0.25):
    """
    评估模型在验证集上的关键点定位精度
    """
    print("=" * 80)
    print("AS-OCT 巩膜突关键点定位精度评估")
    print("=" * 80)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 获取所有验证图片
    val_images_path = Path(val_images_dir)
    val_labels_path = Path(val_labels_dir)

    image_files = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))

    if len(image_files) == 0:
        print(f"错误: 未找到验证图片在 {val_images_dir}")
        return

    print(f"\n找到 {len(image_files)} 张验证图片")
    print(f"置信度阈值: {conf_threshold}")
    print("\n开始评估...\n")

    # 统计数据
    stats = {
        'left_scleral_spur': [],
        'right_scleral_spur': [],
        'all': []
    }

    keypoint_names = ['left_scleral_spur', 'right_scleral_spur']

    no_detection_count = 0
    processed_count = 0

    # 逐张图片评估
    for img_path in image_files:
        # 获取对应的标注文件
        label_name = img_path.stem + '.txt'
        label_path = val_labels_path / label_name

        if not label_path.exists():
            continue

        # 读取图片尺寸
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]

        # 加载真实标注
        gt_keypoints = load_ground_truth_from_txt(label_path, img_width, img_height)

        if gt_keypoints is None:
            continue

        # 模型预测
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        # 提取预测的关键点
        if len(results) == 0 or len(results[0].boxes) == 0:
            no_detection_count += 1
            continue

        result = results[0]

        # 获取第一个检测框的关键点
        if result.keypoints is None or len(result.keypoints) == 0:
            no_detection_count += 1
            continue

        pred_keypoints_norm = result.keypoints.xy[0].cpu().numpy()  # [2, 2]

        # 转换为像素坐标
        pred_keypoints = []
        for kp in pred_keypoints_norm:
            x, y = kp
            pred_keypoints.append((float(x), float(y)))

        # 计算每个关键点的距离
        for i, (pred_kp, gt_kp, kp_name) in enumerate(zip(pred_keypoints, gt_keypoints, keypoint_names)):
            if gt_kp is not None:
                distance = calculate_pixel_distance(pred_kp, gt_kp)

                if distance is not None:
                    stats[kp_name].append(distance)
                    stats['all'].append(distance)

        processed_count += 1

        if processed_count % 20 == 0:
            print(f"  已处理: {processed_count}/{len(image_files)}")

    # 打印结果
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

    print(f"\n处理统计:")
    print(f"  总图片数: {len(image_files)}")
    print(f"  成功检测: {processed_count}")
    print(f"  未检测到: {no_detection_count}")
    print(f"  检测率: {processed_count / len(image_files) * 100:.1f}%")

    # 计算并打印每个关键点的统计
    print("\n" + "-" * 80)
    print("关键点定位精度 (像素距离):")
    print("-" * 80)

    for kp_name in keypoint_names + ['all']:
        distances = stats[kp_name]

        if len(distances) == 0:
            continue

        distances = np.array(distances)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        median_dist = np.median(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # PCK@Xpx (Percentage of Correct Keypoints)
        pck_5 = np.sum(distances <= 5) / len(distances) * 100
        pck_10 = np.sum(distances <= 10) / len(distances) * 100
        pck_20 = np.sum(distances <= 20) / len(distances) * 100
        pck_50 = np.sum(distances <= 50) / len(distances) * 100

        display_name = "整体" if kp_name == 'all' else kp_name

        print(f"\n{display_name}:")
        print(f"  样本数量: {len(distances)}")
        print(f"  平均距离 (MPD): {mean_dist:.2f} ± {std_dist:.2f} 像素")
        print(f"  中位数距离: {median_dist:.2f} 像素")
        print(f"  最小距离: {min_dist:.2f} 像素")
        print(f"  最大距离: {max_dist:.2f} 像素")
        print(f"  PCK@5px:  {pck_5:.1f}%")
        print(f"  PCK@10px: {pck_10:.1f}%")
        print(f"  PCK@20px: {pck_20:.1f}%")
        print(f"  PCK@50px: {pck_50:.1f}%")

    # 绘制距离分布图
    print("\n生成可视化图表...")
    plot_distance_distribution(stats, keypoint_names)

    return stats


def plot_distance_distribution(stats, keypoint_names):
    """
    绘制距离分布图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AS-OCT Scleral Spur Keypoint Localization Error Distribution',
                 fontsize=14, fontweight='bold')

    # 1. 整体距离分布直方图
    ax1 = axes[0, 0]
    all_distances = np.array(stats['all'])
    ax1.hist(all_distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(all_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_distances):.2f}px')
    ax1.axvline(np.median(all_distances), color='green', linestyle='--',
                label=f'Median: {np.median(all_distances):.2f}px')
    ax1.set_xlabel('Pixel Distance (px)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Distance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 左右关键点对比箱线图
    ax2 = axes[0, 1]
    data_to_plot = [stats[kp] for kp in keypoint_names if len(stats[kp]) > 0]
    labels = [kp.replace('_', ' ').title() for kp in keypoint_names if len(stats[kp]) > 0]
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_ylabel('Pixel Distance (px)')
    ax2.set_title('Left vs Right Scleral Spur')
    ax2.grid(True, alpha=0.3)

    # 3. PCK曲线
    ax3 = axes[1, 0]
    thresholds = np.arange(0, 101, 1)

    for kp_name in keypoint_names:
        if len(stats[kp_name]) == 0:
            continue

        distances = np.array(stats[kp_name])
        pck_values = [np.sum(distances <= t) / len(distances) * 100 for t in thresholds]

        label = kp_name.replace('_', ' ').title()
        ax3.plot(thresholds, pck_values, marker='o', markersize=2, label=label)

    ax3.set_xlabel('Distance Threshold (px)')
    ax3.set_ylabel('PCK (%)')
    ax3.set_title('PCK Curve (Percentage of Correct Keypoints)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 105])

    # 4. 累积分布函数 (CDF)
    ax4 = axes[1, 1]
    for kp_name in keypoint_names + ['all']:
        if len(stats[kp_name]) == 0:
            continue

        distances = np.array(sorted(stats[kp_name]))
        cdf = np.arange(1, len(distances) + 1) / len(distances) * 100

        label = "Overall" if kp_name == 'all' else kp_name.replace('_', ' ').title()
        ax4.plot(distances, cdf, label=label, linewidth=2)

    ax4.set_xlabel('Pixel Distance (px)')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Distribution Function (CDF)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 100])

    plt.tight_layout()

    # 保存图表
    save_path = 'runs/pose/evaluation_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图表保存至: {save_path}")

    plt.close()


def main():
    # 配置
    model_path = 'runs/pose/asoct_yolo11x/weights/best.pt'
    val_images_dir = 'datasets/ASOCT_YOLO/images/val'
    val_labels_dir = 'datasets/ASOCT_YOLO/labels/val'
    conf_threshold = 0.25  # 置信度阈值

    # 检查路径
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    if not Path(val_images_dir).exists():
        print(f"错误: 验证集图片目录不存在 - {val_images_dir}")
        print("请先运行 convert_asoct_to_yolo_pose.py 转换数据")
        return

    if not Path(val_labels_dir).exists():
        print(f"错误: 验证集标注目录不存在 - {val_labels_dir}")
        print("请先运行 convert_asoct_to_yolo_pose.py 转换数据")
        return

    # 执行评估
    stats = evaluate_model(
        model_path=model_path,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        conf_threshold=conf_threshold
    )

    print("\n" + "=" * 80)
    print("评估指标说明:")
    print("=" * 80)
    print("1. MPD (Mean Pixel Distance): 平均像素距离，越小越好")
    print("2. STD (Standard Deviation): 标准差，反映稳定性")
    print("3. Median Distance: 中位数距离，对异常值更鲁棒")
    print("4. PCK@Xpx: X像素范围内正确关键点的百分比")
    print("   - PCK@5px: 临床高精度要求")
    print("   - PCK@10px: 临床一般精度要求")
    print("   - PCK@20px: 临床可接受精度")
    print("\n医学应用标准:")
    print("  优秀: MPD < 5px, PCK@10px > 95%")
    print("  良好: MPD < 10px, PCK@20px > 90%")
    print("  可用: MPD < 20px, PCK@50px > 85%")


if __name__ == '__main__':
    main()
