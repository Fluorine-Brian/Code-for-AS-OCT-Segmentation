"""
YOLOv11 Pose 预测脚本
用于对AS-OCT图像进行巩膜突关键点检测
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    print("=" * 70)
    print("YOLOv11 Pose 图像预测 - AS-OCT 巩膜突检测")
    print("=" * 70)

    # 模型路径 (修改为你训练好的模型路径)
    model_path = 'runs/pose/asoct_yolo11s/weights/best.pt'

    # 要预测的图片路径 (可以是文件、文件夹或URL)
    # 示例:
    # - 单个图片: 'datasets/ASOCT_YOLO/images/val/cataract_xxx.jpg'
    # - 文件夹: 'datasets/ASOCT_YOLO/images/val/'
    # - 原始数据: 'datasets/Cataract/Original Images/'
    # - 视频: 'path/to/video.mp4'
    # - 摄像头: 0 (默认摄像头)
    source = 'datasets/ASOCT_YOLO/images/val/'  # 默认预测验证集

    print(f"\n加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"\n错误: 模型文件不存在 - {model_path}")
        print("请先运行 train_asoct_pose.py 训练模型")
        return

    # 检查设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 预测配置
    print("\n预测配置:")
    print("-" * 70)

    config = {
        'source': source,                       # 输入源
        'imgsz': 640,                           # 输入图片大小
        'conf': 0.25,                           # 置信度阈值 (可调整 0.1-0.9)
        'iou': 0.7,                             # NMS IOU阈值
        'device': device,                       # 设备
        'max_det': 300,                         # 最大检测数
        'half': False,                          # FP16推理
        'show': False,                          # 显示结果 (GUI环境)
        'save': True,                           # 保存预测结果
        'save_txt': True,                       # 保存文本格式结果
        'save_conf': True,                      # 在文本中保存置信度
        'save_crop': False,                     # 保存裁剪的检测框
        'show_labels': True,                    # 显示标签
        'show_conf': True,                      # 显示置信度
        'show_boxes': True,                     # 显示边界框
        'line_width': None,                     # 边界框线宽 (None=自动)
        'visualize': False,                     # 可视化特征图
        'augment': False,                       # 测试时增强
        'agnostic_nms': False,                  # 类别无关NMS
        'retina_masks': False,                  # 高分辨率mask
        'project': 'runs/pose',                 # 项目保存路径
        'name': 'predict',                      # 实验名称
        'exist_ok': True,                       # 是否覆盖已存在的实验
        'verbose': True,                        # 详细输出
    }

    for key, value in config.items():
        if key != 'source' or len(str(value)) < 80:
            print(f"  {key:20s}: {value}")

    print("-" * 70)
    print(f"\n预测输入: {source}")
    print("开始预测...\n")

    # 执行预测
    try:
        results = model.predict(**config)

        print("\n" + "=" * 70)
        print("预测完成!")
        print("=" * 70)

        # 统计结果
        total_images = len(results)
        total_detections = sum(len(r.boxes) for r in results)

        print(f"\n处理图片数: {total_images}")
        print(f"总检测数: {total_detections}")
        print(f"平均每张: {total_detections/total_images:.2f} 个检测" if total_images > 0 else "")

        # 找到保存路径
        if results and hasattr(results[0], 'save_dir'):
            save_dir = results[0].save_dir
            print(f"\n结果保存在: {save_dir}")
            print("  - 可视化图片: 带关键点标注的图片")
            print("  - labels/: 文本格式标注")

        print("\n提示:")
        print("- 如需调整检测灵敏度,修改 conf 参数 (0.1-0.9)")
        print("- 如需处理单张图片,修改 source 参数")
        print("- 可视化结果在输出目录中查看")

        # 显示第一张图的关键点信息
        if results and len(results) > 0 and hasattr(results[0], 'keypoints'):
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                print("\n第一张图片的关键点检测:")
                print("-" * 70)
                try:
                    kpts = results[0].keypoints.xy[0]  # 第一个检测框的关键点
                    conf = results[0].boxes.conf[0] if len(results[0].boxes) > 0 else 0.0
                    kpt_names = ['left_scleral_spur', 'right_scleral_spur']

                    print(f"  检测置信度: {conf:.3f}")
                    for i, (kpt, name) in enumerate(zip(kpts, kpt_names)):
                        x, y = kpt
                        print(f"  {name:25s}: ({x:.1f}, {y:.1f})")
                except:
                    pass

        print("\n医学应用:")
        print("- 巩膜突定位用于房角评估")
        print("- 辅助青光眼诊断")
        print("- 白内障手术规划")

    except Exception as e:
        print(f"\n预测出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
