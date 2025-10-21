# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from net import UNet
from data import BinarySegmentationDataset

# 1. 数据路径设置
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'
TEST_DIR = '../data/test'

# 2. 超参数和设备设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 0

# 3. 数据预处理
# 图像变换
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图归一化
])
# 掩码变换
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])


# 4. 定义损失函数 (Dice + BCE)
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce


# 5. 训练和验证函数
def train_one_epoch(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, desc="Training")
    model.train()
    total_loss = 0

    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    dice_score = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            # 计算Dice Score
            preds_binary = (outputs > 0.5).float()
            dice_score += (1 - dice_loss(preds_binary, masks))

    avg_loss = total_loss / len(loader)
    avg_dice = dice_score / len(loader)
    print(f'Validation -> Avg Loss: {avg_loss:.4f}, Avg Dice Score: {avg_dice:.4f}')
    return avg_loss


# 6. 主函数
def main():
    # 创建数据集和加载器
    train_dataset = BinarySegmentationDataset(TRAIN_DIR, image_transform, mask_transform)
    val_dataset = BinarySegmentationDataset(VAL_DIR, image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 初始化模型 (n_channels=1, n_classes=1 for binary)
    model = UNet(n_channels=1, n_classes=1, n_filters=32).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 早停参数
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    # 开始训练
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, combined_loss)
        val_loss = validate(val_loader, model, combined_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_anterior_chamber_model.pth')
            print("Model saved to best_anterior_chamber_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


if __name__ == '__main__':
    main()
