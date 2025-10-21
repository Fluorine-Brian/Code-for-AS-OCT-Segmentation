# predict.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from net import UNet

# 1. 参数设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./best_anterior_chamber_model.pth"
TEST_IMAGE_DIR = "../data/test/images/"
TEST_MASK_DIR = "../data/test/masks/"


# IoU 和 Dice Score 计算函数
def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """
    计算单个掩码的 IoU 和 Dice Score.
    Args:
        pred_mask (np.array): 预测的二值掩码 (0或1).
        true_mask (np.array): 真实的二值掩码 (0或1).
        smooth (float): 防止除以零的平滑系数.
    Returns:
        iou (float): IoU 分数.
        dice (float): Dice 分数.
    """
    # 确保掩码是布尔类型或整数类型
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    # 计算交集和并集
    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask | true_mask)

    # 计算 IoU
    iou = (intersection + smooth) / (union + smooth)

    # 计算 Dice Score
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    dice = (2. * intersection + smooth) / (np.sum(pred_mask) + np.sum(true_mask) + smooth)

    return iou, dice


# 2. 主函数
def main():
    # 加载模型
    model = UNet(n_channels=1, n_classes=1, n_filters=32).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # 随机选择几张测试图片
    test_images = os.listdir(TEST_IMAGE_DIR)
    sample_images = random.sample(test_images, min(5, len(test_images)))

    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 掩码预处理
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # 用于存储所有样本的分数
    iou_scores = []
    dice_scores = []

    # 可视化
    plt.figure(figsize=(15, len(sample_images) * 5))

    for i, img_name in enumerate(sample_images):
        # --- 加载图像 ---
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)
        image_pil = Image.open(img_path).convert("L")

        # 加载对应的真实掩码
        # 假设掩码和图像的文件名相同，只是扩展名不同
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(TEST_MASK_DIR, mask_name)
        true_mask_pil = Image.open(mask_path).convert("L")

        # 预处理图像以输入模型
        input_tensor = image_transform(image_pil).unsqueeze(0).to(DEVICE)

        # 预处理真实掩码用于比较
        true_mask_tensor = mask_transform(true_mask_pil)
        true_mask_np = (true_mask_tensor > 0.5).squeeze().numpy()

        # 模型预测
        with torch.no_grad():
            output = model(input_tensor)

        # 将概率图 > 0.5 的部分视为前景
        pred_mask_np = (output > 0.5).float().squeeze(0).cpu().squeeze().numpy()

        # 计算并记录当前样本的评估指标
        iou, dice = calculate_metrics(pred_mask_np, true_mask_np)
        iou_scores.append(iou)
        dice_scores.append(dice)

        print(f"Image: {img_name} -> IoU: {iou:.4f}, Dice: {dice:.4f}")

        # --- 可视化部分 ---
        # 显示原图
        plt.subplot(len(sample_images), 3, 3 * i + 1)
        plt.imshow(image_pil, cmap='gray')
        plt.title(f"Original: {img_name}")
        plt.axis('off')

        # 显示预测掩码
        plt.subplot(len(sample_images), 3, 3 * i + 2)
        plt.imshow(pred_mask_np, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        # 显示叠加结果，并在标题中显示分数
        plt.subplot(len(sample_images), 3, 3 * i + 3)
        plt.imshow(image_pil, cmap='gray')
        plt.imshow(pred_mask_np, cmap='Reds', alpha=0.4)
        plt.title(f"Overlay\nIoU: {iou:.4f} | Dice: {dice:.4f}")
        plt.axis('off')

    # --- 新增: 计算并打印所有样本的平均分数 ---
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    print("\n" + "=" * 30)
    print(f"Average Metrics over {len(sample_images)} samples:")
    print(f"Average IoU:  {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print("=" * 30)

    # 在总标题中显示平均分
    plt.suptitle(f'Model Predictions (Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应总标题
    plt.savefig('./predictions_anterior_chamber.png')
    plt.show()
    print("Visualization saved as predictions_anterior_chamber.png")


if __name__ == '__main__':
    main()
