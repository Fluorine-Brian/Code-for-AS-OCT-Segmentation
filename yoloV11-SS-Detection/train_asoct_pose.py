"""
YOLOv11 Pose 训练脚本
用于AS-OCT (Anterior Segment OCT) 图像的巩膜突关键点检测
支持4个类别合并训练: Cataract, Normal, PACG, PACG_Cataract
"""

from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("YOLOv11 Pose 训练 - AS-OCT 巩膜突关键点检测")
    print("=" * 70)

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        device = 0
    else:
        print("\n警告: CUDA不可用，使用CPU训练 (速度会很慢)")
        device = 'cpu'

    # 加载预训练模型
    # 可选模型: yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt, yolo11l-pose.pt, yolo11x-pose.pt
    # n(nano) < s(small) < m(medium) < l(large) < x(xlarge)
    # 推荐: 开始用n或s快速验证，效果好再用m或l
    model_name = 'yolo11x-pose.pt'  # 使用large模型，平衡速度和精度
    print(f"\n加载模型: {model_name}")

    model = YOLO(model_name)

    # 训练参数
    print("\n训练配置:")
    print("-" * 70)

    config = {
        'data': 'datasets/asoct-pose.yaml',    # 数据集配置文件
        'epochs': 150,                          # 训练轮数 (数据较多，增加轮数)
        'imgsz': 640,                           # 输入图片大小 (会自动resize)
        'batch': 16,                            # 批次大小 (根据GPU显存调整)
        'device': device,                       # GPU设备
        'workers': 8,                           # 数据加载线程数
        'project': 'runs/pose',                 # 项目保存路径
        'name': 'asoct_yolo11x',                # 实验名称
        'exist_ok': False,                      # 是否覆盖已存在的实验
        'pretrained': True,                     # 使用预训练权重
        'optimizer': 'auto',                    # 优化器 ('SGD', 'Adam', 'AdamW', 'auto')
        'lr0': 0.01,                            # 初始学习率
        'lrf': 0.01,                            # 最终学习率 (lr0 * lrf)
        'momentum': 0.937,                      # SGD动量/Adam beta1
        'weight_decay': 0.0005,                 # 权重衰减
        'warmup_epochs': 3.0,                   # 预热轮数
        'warmup_momentum': 0.8,                 # 预热初始动量
        'warmup_bias_lr': 0.1,                  # 预热偏置学习率
        'box': 0,                               # 边界框损失权重
        'cls': 1,                               # 分类损失权重
        'pose': 15.0,                           # 关键点损失权重 (重要!)
        'kobj': 1.0,                            # 关键点目标损失权重
        'label_smoothing': 0.0,                 # 标签平滑
        'save': True,                           # 保存检查点
        'save_period': -1,                      # 每N轮保存一次 (-1只保存最后/最佳)
        'val': True,                            # 训练期间验证
        'plots': True,                          # 保存训练曲线和预测可视化
        'patience': 50,                         # 早停耐心值 (验证无改善停止训练)
        'resume': False,                        # 是否从上次中断处恢复训练
        'amp': True,                            # 自动混合精度训练 (加速训练)
        'fraction': 1.0,                        # 使用数据集的比例 (1.0=100%)
        'profile': False,                       # 性能分析
        'freeze': None,                         # 冻结层数 (None=不冻结)
        'multi_scale': False,                   # 多尺度训练
        'single_cls': True,                     # 单类训练 (我们只有一个类别)
        'rect': False,                          # 矩形训练 (减少padding)
        'cos_lr': False,                        # 余弦学习率调度
        'close_mosaic': 10,                     # 最后N轮关闭mosaic增强
        'overlap_mask': True,                   # 训练时mask可重叠
        'mask_ratio': 4,                        # mask下采样率
        'dropout': 0.0,                         # Dropout率 (仅分类)
        'cache': False,                         # 缓存图片到内存/硬盘 (True/'ram'/'disk')
        'verbose': True,                        # 详细输出
    }

    # 打印配置
    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    print("-" * 70)
    print(f"\n数据集信息:")
    print(f"  总样本数: ~717 张")
    print(f"  训练集: ~574 张 (80%)")
    print(f"  验证集: ~143 张 (20%)")
    print(f"  类别: 1 (asoct - 合并所有原始类别)")
    print(f"  关键点: 2 (left_scleral_spur, right_scleral_spur)")

    print("\n原始类别分布:")
    print(f"  Cataract: 380 张 (53.0%)")
    print(f"  Normal: 50 张 (7.0%)")
    print(f"  PACG: 148 张 (20.6%)")
    print(f"  PACG_Cataract: 139 张 (19.4%)")

    print("\n开始训练...")
    print("提示: 训练过程中可以按 Ctrl+C 安全停止\n")

    # 开始训练
    try:
        results = model.train(**config)

        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"\n最佳模型: {config['project']}/{config['name']}/weights/best.pt")
        print(f"最后模型: {config['project']}/{config['name']}/weights/last.pt")
        print(f"训练曲线: {config['project']}/{config['name']}/")

        print("\n下一步:")
        print("1. 查看训练曲线: results.png")
        print("2. 验证模型: python validate_asoct_pose.py")
        print("3. 预测图片: python predict_asoct_pose.py")
        print("4. 导出模型: yolo export model=runs/pose/asoct_yolo11s/weights/best.pt format=onnx")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print(f"已保存的检查点: {config['project']}/{config['name']}/weights/")
        print("可以使用 resume=True 继续训练")

    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        print("\n常见问题解决:")
        print("1. GPU内存不足: 减小 batch 大小 (如: batch=8 或 batch=4)")
        print("2. 数据集路径错误: 检查 asoct-pose.yaml 中的路径")
        print("3. 数据未转换: 先运行 python convert_asoct_to_yolo_pose.py")
        print("4. 依赖缺失: pip install ultralytics torch torchvision scikit-learn")


if __name__ == '__main__':
    main()
