import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from operator import add
from tqdm import tqdm

from model import TResUnet
from utils import calculate_metrics, create_dir


# 沿用训练脚本中的 Dataset 类
class OCTDataset(Dataset):
    def __init__(self, base_folder, image_subfolder, mask_subfolder, size):
        """
        自定义数据集类，用于加载OCT图像及其对应的单个掩码。
        用于评估时，返回图像和掩码的路径以及处理后的 tensor。

        Args:
            base_folder (str): 数据集根目录 (例如 'path/to/dataset_split/test')
            image_subfolder (str): 图像子文件夹名称 (例如 'images')
            mask_subfolder (str): 掩码子文件夹名称 (例如 'anterior_chamber_masks')
            size (tuple): 图像和掩码将被resize到的尺寸 (width, height)
        """
        super().__init__()

        # 使用 glob 获取图像路径，并按文件名排序以确保与掩码对应
        self.image_paths = sorted(glob(os.path.join(base_folder, image_subfolder, "*")))
        self.mask_paths = []

        # 假设掩码文件名与原始图像文件名（不含扩展名）一致，且掩码扩展名为 .png
        for img_path in self.image_paths:
            img_filename = os.path.basename(img_path)
            base_filename = os.path.splitext(img_filename)[0]
            mask_filename = base_filename + '.png'
            mask_path = os.path.join(base_folder, mask_subfolder, mask_filename)
            self.mask_paths.append(mask_path)

        self.size = size
        self.n_samples = len(self.image_paths)

        # 检查图像和掩码数量是否匹配
        if len(self.image_paths) != len(self.mask_paths):
            print(
                f"警告: 图像数量 ({len(self.image_paths)}) 与掩码数量 ({len(self.mask_paths)}) 不匹配在文件夹 {base_folder}/{mask_subfolder}")
            # 如果数量不匹配，可以进一步检查哪些文件缺失，或者直接过滤掉不匹配的对

    def __getitem__(self, index):
        """ Image """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # 读取图像 (灰度)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            # 返回一个占位符，或者跳过，这里返回零数组作为示例
            # 注意cv2尺寸是(width, height)，numpy是(height, width)
            image = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
            mask = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        else:
            # 读取掩码 (灰度)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码文件 {mask_path}，使用全黑掩码代替。")
                mask = np.zeros_like(image, dtype=np.uint8)  # 使用与图像相同尺寸的零数组

        # 将灰度图像转换为3通道，以匹配模型输入的期望
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Resize 图像和掩码
        # cv2.resize 期望 (width, height)
        image_resized = cv2.resize(image_rgb, self.size)
        mask_resized = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)  # 掩码使用最近邻插值，避免引入中间值

        # 转换为 PyTorch Tensor 格式 (C, H, W) 并归一化
        # 图像: (H, W, C) -> (C, H, W), 归一化到 [0, 1]
        image_tensor = np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0

        # 掩码: (H, W) -> (1, H, W), 归一化到 [0, 1] (二值掩码 0 或 1)
        # 确保掩码是二值的 (0 或 1)，即使resize可能引入中间值
        mask_tensor = (mask_resized > 127).astype(np.float32)  # 阈值化为二值
        mask_tensor = np.expand_dims(mask_tensor, axis=0)  # 添加通道维度

        # 返回处理后的 tensor 以及原始文件路径
        return image_tensor, mask_tensor, image_path, mask_path

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    """ Device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备进行评估: {device}")
    if torch.cuda.is_available():
        print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")

    """ Hyperparameters (应与训练时一致) """
    image_size = 256  # 模型输入尺寸
    size = (image_size, image_size)  # (width, height)
    batch_size = 16

    # 数据集根路径 (包含 train 和 test 文件夹)
    dataset_root_path = r"C:\srp_OCT\TransResUNet-main\AS-OCT_Segmentation\MedSAM\data"  # 改为自己的路径

    # 定义需要评估的掩码类型及其对应的子文件夹名称
    mask_types_to_evaluate = {
        'anterior_chamber': 'anterior_chamber',
        'lens': 'lens',
        'nucleus': 'nucleus',
        'right_iris': 'right_iris',
        'left_iris': 'left_iris'
    }
    image_subfolder_name = 'images'  # 原始图像子文件夹名称

    # 训练好的模型检查点所在的目录
    checkpoints_base_dir = "files_tresunet_segmentation"

    # 可视化结果保存的根目录
    output_results_base_dir = "evaluation_results_viz"  # 修改保存目录名称以区分
    create_dir(output_results_base_dir)

    # 定义每种掩码类型的可视化颜色 (BGR 格式, 0-255)
    # 尽量使用浅色，不要太鲜艳
    mask_colors = {
        'anterior_chamber': (200, 200, 100),  # 浅黄色
        'lens': (100, 200, 100),  # 浅绿色
        'nucleus': (100, 100, 200),  # 浅蓝色
        'left_iris': (200, 100, 200),  # 浅紫色
        'right_iris': (200, 100, 200)
        # 可以根据需要调整这些颜色
    }

    print("\n--- 开始评估模型 ---")

    # 数据集路径
    test_base_folder = os.path.join(dataset_root_path, 'test')

    # --- 循环评估每种掩码类型 ---
    for mask_type, mask_subfolder_name in mask_types_to_evaluate.items():
        print(f"\n--- 评估 {mask_type} 的分割模型 ---")

        # 检查点文件路径
        checkpoint_path = os.path.join(checkpoints_base_dir, f"checkpoint_{mask_type}.pth")

        # 检查检查点文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"错误: 未找到 {mask_type} 的模型检查点文件: {checkpoint_path}，跳过评估。")
            continue

        # 获取当前掩码类型的可视化颜色
        current_mask_color = mask_colors.get(mask_type, (128, 128, 128))  # 如果未定义颜色，使用灰色

        # 创建测试数据集实例
        test_dataset = OCTDataset(test_base_folder, image_subfolder_name, mask_subfolder_name, size)

        # 检查测试集是否为空
        if len(test_dataset) == 0:
            print(f"警告: {mask_type} 的测试集为空，无法进行评估。")
            continue

        # 创建 DataLoader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  # 评估时不需要打乱
            num_workers=2  # 根据你的机器性能调整 num_workers
        )

        # --- 创建可视化结果保存目录 ---
        mask_results_dir = os.path.join(output_results_base_dir, mask_type)
        joint_save_dir = os.path.join(mask_results_dir, "joint")
        mask_save_dir = os.path.join(mask_results_dir, "predicted_mask")  # 保存预测的二值掩码
        overlay_save_dir = os.path.join(mask_results_dir, "overlay")  # 保存叠加图

        create_dir(joint_save_dir)
        create_dir(mask_save_dir)
        create_dir(overlay_save_dir)
        print(f"可视化结果将保存在: {mask_results_dir}")

        """ Model """
        model = TResUnet()  # 实例化模型结构
        # 加载训练好的权重
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()  # 设置模型为评估模式 (关闭 dropout, batchnorm 等的训练行为)
            print(f"成功加载模型权重: {checkpoint_path}")
        except Exception as e:
            print(f"错误: 加载模型权重失败 {checkpoint_path}: {e}，跳过评估。")
            continue

        """ Evaluation Loop """
        # --- 修改: 存储每个样本的指标，而不是直接累加批次平均值 ---
        # 假设 calculate_metrics 返回 Jaccard, F1, Recall, Precision, Accuracy, F2, 共 6 个指标
        all_jaccard = []
        all_f1 = []
        all_recall = []
        all_precision = []
        all_accuracy = []
        all_f2 = []

        time_taken = []  # 用于计算FPS

        num_batches = len(test_loader)

        if num_batches == 0:
            print(f"警告: {mask_type} 的测试集 DataLoader 为空，无法进行评估。")
            continue

        with torch.no_grad():  # 在评估时不需要计算梯度，可以节省内存和计算
            for i, (x, y, img_paths, mask_paths) in tqdm(enumerate(test_loader), total=num_batches,
                                                         desc=f"Evaluating {mask_type}"):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                # --- FPS calculation ---
                start_time = time.time()
                y_pred = model(x)
                end_time = time.time() - start_time
                time_taken.append(end_time)

                y_pred_prob = torch.sigmoid(y_pred)
                y_pred_binary = (y_pred_prob > 0.5).float()

                # --- 可视化保存 (保持不变) ---
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                y_pred_binary_np = y_pred_binary.cpu().numpy()

                batch_size_actual = x.size(0)

                for j in range(batch_size_actual):
                    img_filename = os.path.basename(img_paths[j])
                    name = os.path.splitext(img_filename)[0]

                    original_img_viz = np.transpose(x_np[j], (1, 2, 0)) * 255.0
                    original_img_viz = original_img_viz.astype(np.uint8)

                    gt_mask_viz = np.squeeze(y_np[j], axis=0) * 255.0
                    gt_mask_viz = gt_mask_viz.astype(np.uint8)
                    gt_mask_viz = np.stack([gt_mask_viz] * 3, axis=-1)

                    pred_mask_viz = np.squeeze(y_pred_binary_np[j], axis=0) * 255.0
                    pred_mask_viz = pred_mask_viz.astype(np.uint8)
                    pred_mask_viz_3channel = np.stack([pred_mask_viz] * 3, axis=-1)

                    mask_bool = pred_mask_viz > 0
                    overlay_img_viz = original_img_viz.copy()
                    overlay_img_viz[mask_bool] = current_mask_color

                    line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

                    cat_images = np.concatenate([
                        original_img_viz, line,
                        gt_mask_viz, line,
                        pred_mask_viz_3channel, line,
                        overlay_img_viz
                    ], axis=1)

                    cv2.imwrite(os.path.join(joint_save_dir, f"{name}.jpg"), cat_images)
                    cv2.imwrite(os.path.join(mask_save_dir, f"{name}.png"), pred_mask_viz)
                    cv2.imwrite(os.path.join(overlay_save_dir, f"{name}.jpg"), overlay_img_viz)

                # --- 计算指标 (修改为存储每个样本的指标) ---
                for j in range(batch_size_actual):
                    single_y_true = y[j]
                    single_y_pred_binary = y_pred_binary[j]

                    score = calculate_metrics(single_y_true, single_y_pred_binary)

                    # 假设 calculate_metrics 返回顺序是 Jaccard, F1, Recall, Precision, Accuracy, F2
                    all_jaccard.append(score[0])
                    all_f1.append(score[1])
                    all_recall.append(score[2])
                    all_precision.append(score[3])
                    all_accuracy.append(score[4])
                    all_f2.append(score[5])

        # --- 计算均值和标准差 ---
        # 确保列表不为空，避免计算 np.mean/np.std 时报错
        if all_jaccard:
            mean_jaccard, std_jaccard = np.mean(all_jaccard), np.std(all_jaccard)
            mean_f1, std_f1 = np.mean(all_f1), np.std(all_f1)
            mean_recall, std_recall = np.mean(all_recall), np.std(all_recall)
            mean_precision, std_precision = np.mean(all_precision), np.std(all_precision)
            mean_accuracy, std_accuracy = np.mean(all_accuracy), np.std(all_accuracy)
            mean_f2, std_f2 = np.mean(all_f2), np.std(all_f2)
        else:
            mean_jaccard, std_jaccard = 0.0, 0.0
            mean_f1, std_f1 = 0.0, 0.0
            mean_recall, std_recall = 0.0, 0.0
            mean_precision, std_precision = 0.0, 0.0
            mean_accuracy, std_accuracy = 0.0, 0.0
            mean_f2, std_f2 = 0.0, 0.0

        # 计算平均FPS
        mean_time_per_image = np.mean(time_taken) / batch_size if time_taken and batch_size > 0 else 0
        mean_fps = 1 / mean_time_per_image if mean_time_per_image > 0 else 0

        print(f"\n{mask_type} 测试集评估结果:")
        print(f"  IoU (Jaccard): {mean_jaccard:.4f} \u00B1 {std_jaccard:.4f}")  # IoU通常指Jaccard
        print(f"  Dice (F1):     {mean_f1:.4f} \u00B1 {std_f1:.4f}")
        print(f"  Recall:        {mean_recall:.4f} \u00B1 {std_recall:.4f}")
        print(f"  Precision:     {mean_precision:.4f} \u00B1 {std_precision:.4f}")
        print(f"  Accuracy:      {mean_accuracy:.4f} \u00B1 {std_accuracy:.4f}")
        print(f"  F2 Score:      {mean_f2:.4f} \u00B1 {std_f2:.4f}")
        print(f"  平均推理时间 (每张图像): {mean_time_per_image:.4f} 秒")
        print(f"  平均 FPS: {mean_fps:.2f}")

    print("\n所有掩码类型的模型评估完成。")

