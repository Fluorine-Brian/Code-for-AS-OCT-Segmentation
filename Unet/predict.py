# predict.py (已修改，包含均值和标准差输出)

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm  # 引入tqdm用于显示进度条

from net import UNet

# 1. 参数设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['anterior_chamber', 'lens', 'nucleus', 'left_iris', 'right_iris']
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "./best_multiclass_model.pth"
TEST_DIR = '../dataset/segmentation_dataset/test'
TEST_IMAGE_DIR = os.path.join(TEST_DIR, "images")


def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """
    计算单个掩码的 IoU, Dice, Precision, 和 Recall.
    Args:
        pred_mask (np.array): 预测的二值掩码 (0或1).
        true_mask (np.array): 真实的二值掩码 (0或1).
        smooth (float): 防止除以零的平滑系数.
    Returns:
        iou, dice, precision, recall (float): 四个评估指标.
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.sum(pred_mask & true_mask)
    pred_sum = np.sum(pred_mask)
    true_sum = np.sum(true_mask)
    union = np.sum(pred_mask | true_mask)

    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (pred_sum + true_sum + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (true_sum + smooth)

    return iou, dice, precision, recall


# 2. 主函数
def main():
    # 加载模型
    model = UNet(n_channels=1, n_classes=NUM_CLASSES, n_filters=32).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # 获取所有测试图片
    test_images = os.listdir(TEST_IMAGE_DIR)
    if not test_images:
        print(f"错误: 在 '{TEST_IMAGE_DIR}' 中没有找到任何测试图片。")
        return

    print(f"在整个测试集 ({len(test_images)} 张图片) 上进行评估...")

    # 图像和掩码的预处理
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # 用于存储所有样本的分数
    all_scores = {cls_name: {'iou': [], 'dice': [], 'precision': [], 'recall': []} for cls_name in CLASSES}
    for img_name in tqdm(test_images, desc="Evaluating Test Set"):
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)
        image_pil = Image.open(img_path).convert("L")
        input_tensor = image_transform(image_pil).unsqueeze(0).to(DEVICE)

        # 1. 获取模型预测
        with torch.no_grad():
            output = model(input_tensor)
        # 预测结果形状为 [NUM_CLASSES, H, W]
        pred_masks_np = (torch.sigmoid(output) > 0.5).float().squeeze(0).cpu().numpy()

        # 2. 加载所有真实掩码并进行评估
        true_masks_list = []
        base_name = os.path.splitext(img_name)[0]

        for i, cls_name in enumerate(CLASSES):
            mask_path = os.path.join(TEST_DIR, cls_name, f"{base_name}.png")
            if not os.path.exists(mask_path):
                # 如果某个真实掩码不存在，创建一个全黑的作为占位符
                true_mask_tensor = torch.zeros((1, 256, 256))
            else:
                true_mask_pil = Image.open(mask_path).convert("L")
                true_mask_tensor = mask_transform(true_mask_pil)

            true_masks_list.append(true_mask_tensor)

            # 分别计算每个类别的指标
            true_mask_np_single = (true_mask_tensor > 0.5).squeeze().numpy()
            pred_mask_np_single = pred_masks_np[i]

            iou, dice, precision, recall = calculate_metrics(pred_mask_np_single, true_mask_np_single)
            all_scores[cls_name]['iou'].append(iou)
            all_scores[cls_name]['dice'].append(dice)
            all_scores[cls_name]['precision'].append(precision)
            all_scores[cls_name]['recall'].append(recall)

    print("\n" + "=" * 60)
    print("在测试集上的各类别平均指标:")
    print("-" * 60)

    overall_metrics = {'dice': [], 'iou': []}
    for cls_name in CLASSES:
        if all_scores[cls_name]['dice']:
            mean_dice = np.mean(all_scores[cls_name]['dice'])
            std_dice = np.std(all_scores[cls_name]['dice'])
            mean_iou = np.mean(all_scores[cls_name]['iou'])
            std_iou = np.std(all_scores[cls_name]['iou'])

            overall_metrics['dice'].append(mean_dice)
            overall_metrics['iou'].append(mean_iou)

            print(
                f"  - Class: {cls_name:<20} | Avg Dice: {mean_dice:.4f} \u00B1 {std_dice:.4f} | Avg IoU: {mean_iou:.4f} \u00B1 {std_iou:.4f}")
        else:
            print(f"  - Class: {cls_name:<20} | No data found.")

    print("-" * 60)
    if overall_metrics['dice']:
        print(f"  Overall Mean Dice Score (mDice): {np.mean(overall_metrics['dice']):.4f}")
        print(f"  Overall Mean IoU (mIoU):       {np.mean(overall_metrics['iou']):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
