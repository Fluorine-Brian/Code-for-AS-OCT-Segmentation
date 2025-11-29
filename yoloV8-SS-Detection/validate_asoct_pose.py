"""
YOLOv11 Pose 验证脚本
用于评估训练好的AS-OCT巩膜突检测模型性能
"""

from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("YOLOv11 Pose 模型验证 - AS-OCT 巩膜突检测")
    print("=" * 70)

    # 模型路径 (修改为你训练好的模型路径)
    model_path = 'runs/pose/asoct_yolo11x/weights/best.pt'

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

    # 验证配置
    print("\n验证配置:")
    print("-" * 70)

    config = {
        'data': 'datasets/asoct-pose.yaml',    # 数据集配置文件
        'imgsz': 640,                           # 输入图片大小
        'batch': 16,                            # 批次大小
        'device': device,                       # 设备
        'workers': 8,                           # 数据加载线程数
        'verbose': True,                        # 详细输出
        'save_json': True,                      # 保存JSON格式结果
        'save_hybrid': False,                   # 保存混合标签
        'conf': 0.001,                          # 置信度阈值
        'iou': 0.6,                             # NMS IOU阈值
        'max_det': 300,                         # 最大检测数
        'half': False,                          # FP16推理
        'plots': True,                          # 保存预测可视化
        'split': 'val',                         # 数据集划分 ('val', 'test')
    }

    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    print("-" * 70)
    print("\n开始验证...")

    # 执行验证
    try:
        metrics = model.val(**config)

        print("\n" + "=" * 70)
        print("验证完成!")
        print("=" * 70)

        # 打印关键指标
        print("\n关键指标:")
        print("-" * 70)
        print(f"  mAP50 (box):    {metrics.box.map50:.4f}")
        print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
        print(f"  mAP50 (pose):   {metrics.pose.map50:.4f}")
        print(f"  mAP50-95 (pose):{metrics.pose.map:.4f}")
        print("-" * 70)

        print("\n指标说明:")
        print("- mAP50: 在IoU=0.5时的平均精度")
        print("- mAP50-95: 在IoU=0.5:0.95时的平均精度 (COCO标准)")
        print("- box: 边界框检测指标")
        print("- pose: 关键点检测指标 (主要关注这个)")

        print(f"\n关键点: left_scleral_spur, right_scleral_spur")
        print(f"验证集: ~143 张图片")

        print("\n结果已保存到模型目录")

    except Exception as e:
        print(f"\n验证出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
