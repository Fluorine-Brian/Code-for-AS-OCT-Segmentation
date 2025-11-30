"""
可视化预测结果和误差
在图片上同时显示预测点、真实点和距离
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch


def load_ground_truth_from_txt(label_path, img_width, img_height):
    """
    从YOLO格式的txt文件加载真实关键点
    """
    if not label_path.exists():
        return None

    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = list(map(float, line.split()))

    if len(parts) < 11:
        return None

    # 提取关键点
    kp1_x_norm, kp1_y_norm, kp1_v = parts[5], parts[6], parts[7]
    kp2_x_norm, kp2_y_norm, kp2_v = parts[8], parts[9], parts[10]

    keypoints = []

    # left_scleral_spur
    if kp1_v > 0:
        x = kp1_x_norm * img_width
        y = kp1_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    # right_scleral_spur
    if kp2_v > 0:
        x = kp2_x_norm * img_width
        y = kp2_y_norm * img_height
        keypoints.append((x, y))
    else:
        keypoints.append(None)

    return keypoints


def visualize_predictions_with_error(model_path, val_images_dir, val_labels_dir,
                                     output_dir='runs/pose/visualize_errors',
                                     conf_threshold=0.25, max_images=20):
    """
    可视化预测结果，显示预测点、真实点和误差
    """
    print("=" * 80)
    print("可视化预测结果和误差")
    print("=" * 80)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 获取验证图片
    val_images_path = Path(val_images_dir)
    val_labels_path = Path(val_labels_dir)

    image_files = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))

    if len(image_files) == 0:
        print(f"错误: 未找到图片在 {val_images_dir}")
        return

    # 限制处理数量
    image_files = image_files[:max_images]

    print(f"\n处理 {len(image_files)} 张图片")
    print(f"输出目录: {output_path.absolute()}\n")

    keypoint_names = ['Left Scleral Spur', 'Right Scleral Spur']
    colors_gt = [(0, 255, 0), (0, 255, 0)]      # 绿色 - 真实标注
    colors_pred = [(255, 0, 0), (255, 0, 0)]    # 蓝色 - 预测结果

    processed = 0

    for img_path in image_files:
        # 获取标注文件
        label_name = img_path.stem + '.txt'
        label_path = val_labels_path / label_name

        if not label_path.exists():
            continue

        # 读取图片
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

        # 创建可视化图像
        vis_img = img.copy()

        # 提取预测关键点
        if len(results) > 0 and len(results[0].boxes) > 0:
            result = results[0]

            if result.keypoints is not None and len(result.keypoints) > 0:
                pred_keypoints_norm = result.keypoints.xy[0].cpu().numpy()

                # 转换为像素坐标
                pred_keypoints = []
                for kp in pred_keypoints_norm:
                    x, y = kp
                    pred_keypoints.append((float(x), float(y)))

                # 获取检测置信度
                conf = float(result.boxes.conf[0].cpu().numpy())

                # 绘制每个关键点
                for i, (pred_kp, gt_kp, kp_name, color_gt, color_pred) in enumerate(
                    zip(pred_keypoints, gt_keypoints, keypoint_names, colors_gt, colors_pred)
                ):
                    if gt_kp is not None:
                        # 计算距离
                        distance = np.sqrt((pred_kp[0] - gt_kp[0])**2 + (pred_kp[1] - gt_kp[1])**2)

                        # 绘制真实点 (绿色圆圈)
                        gt_pt = (int(gt_kp[0]), int(gt_kp[1]))
                        cv2.circle(vis_img, gt_pt, 8, color_gt, 2)
                        cv2.circle(vis_img, gt_pt, 3, color_gt, -1)

                        # 绘制预测点 (蓝色叉)
                        pred_pt = (int(pred_kp[0]), int(pred_kp[1]))
                        cv2.drawMarker(vis_img, pred_pt, color_pred,
                                      markerType=cv2.MARKER_CROSS,
                                      markerSize=15, thickness=2)

                        # 绘制连线
                        cv2.line(vis_img, gt_pt, pred_pt, (0, 255, 255), 2)

                        # 标注距离
                        mid_x = int((gt_kp[0] + pred_kp[0]) / 2)
                        mid_y = int((gt_kp[1] + pred_kp[1]) / 2)

                        text = f"{distance:.1f}px"
                        cv2.putText(vis_img, text, (mid_x + 10, mid_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # 标注关键点名称
                        label_text = f"{kp_name[:4]}"
                        cv2.putText(vis_img, label_text, (gt_pt[0] - 20, gt_pt[1] - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 2)

                # 添加图例和统计信息
                legend_y = 30
                cv2.putText(vis_img, "Legend:", (10, legend_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.circle(vis_img, (150, legend_y - 5), 5, (0, 255, 0), -1)
                cv2.putText(vis_img, "Ground Truth", (165, legend_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.drawMarker(vis_img, (150, legend_y + 25), (255, 0, 0),
                              markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.putText(vis_img, "Prediction", (165, legend_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(vis_img, f"Confidence: {conf:.2f}", (10, legend_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            # 未检测到
            cv2.putText(vis_img, "No Detection", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # 仍然绘制真实标注
            for gt_kp, kp_name, color_gt in zip(gt_keypoints, keypoint_names, colors_gt):
                if gt_kp is not None:
                    gt_pt = (int(gt_kp[0]), int(gt_kp[1]))
                    cv2.circle(vis_img, gt_pt, 8, color_gt, 2)
                    cv2.circle(vis_img, gt_pt, 3, color_gt, -1)
                    cv2.putText(vis_img, kp_name[:4], (gt_pt[0] - 20, gt_pt[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 2)

        # 保存可视化结果
        output_file = output_path / f"vis_{img_path.name}"
        cv2.imwrite(str(output_file), vis_img)

        processed += 1
        print(f"  [{processed}/{len(image_files)}] {img_path.name}")

    print(f"\n完成! 可视化结果保存在: {output_path.absolute()}")
    print(f"处理图片数: {processed}")


def main():
    # 配置
    model_path = 'runs/pose/asoct_yolo11l/weights/best.pt'
    val_images_dir = 'datasets/ASOCT_YOLO/images/val'
    val_labels_dir = 'datasets/ASOCT_YOLO/labels/val'
    output_dir = 'runs/pose/visualize_errors'
    conf_threshold = 0.25
    max_images = 100  # 最多可视化50张图片

    # 检查路径
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    if not Path(val_images_dir).exists():
        print(f"错误: 验证集目录不存在 - {val_images_dir}")
        print("请先运行 convert_asoct_to_yolo_pose.py 转换数据")
        return

    # 执行可视化
    visualize_predictions_with_error(
        model_path=model_path,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        max_images=max_images
    )

    print("\n说明:")
    print("  绿色圆圈: 真实标注 (Ground Truth)")
    print("  蓝色叉号: 模型预测 (Prediction)")
    print("  黄色连线: 预测误差")
    print("  黄色数字: 像素距离")


if __name__ == '__main__':
    main()
