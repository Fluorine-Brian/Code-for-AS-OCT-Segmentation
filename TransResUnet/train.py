import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import TResUnet
from metrics import DiceLoss, DiceBCELoss


class OCTDataset(Dataset):
    def __init__(self, base_folder, image_subfolder, mask_subfolder, size, transform=None):
        """
        自定义数据集类，用于加载OCT图像及其对应的单个掩码。

        Args:
            base_folder (str): 数据集根目录 (例如 'path/to/dataset/train')
            image_subfolder (str): 图像子文件夹名称 (例如 'images')
            mask_subfolder (str): 掩码子文件夹名称 (例如 'anterior_chamber')
            size (tuple): 图像和掩码将被resize到的尺寸 (width, height)
            transform (albumentations.Compose, optional): 数据增强转换. Defaults to None.
        """
        super().__init__()

        self.image_paths = sorted(glob(os.path.join(base_folder, image_subfolder, "*")))
        self.mask_paths = []
        for img_path in self.image_paths:
            img_filename = os.path.basename(img_path)
            base_filename = os.path.splitext(img_filename)[0]
            mask_filename = base_filename + '.png'
            mask_path = os.path.join(base_folder, mask_subfolder, mask_filename)
            self.mask_paths.append(mask_path)

        self.size = size
        self.transform = transform
        self.n_samples = len(self.image_paths)

        if len(self.image_paths) != len(self.mask_paths):
            print(
                f"警告: 在 {base_folder} 中，图像数量 ({len(self.image_paths)}) 与掩码数量 ({len(self.mask_paths)}) 不匹配。")

    def __getitem__(self, index):
        """ Image """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            image = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
            mask = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码文件 {mask_path}，使用全黑掩码代替。")
                mask = np.zeros_like(image, dtype=np.uint8)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.transform is not None:
            augmentations = self.transform(image=image_rgb, mask=mask)
            image_rgb = augmentations["image"]
            mask = augmentations["mask"]

        image_resized = cv2.resize(image_rgb, self.size)
        mask_resized = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        image_tensor = np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0
        mask_tensor = (mask_resized > 127).astype(np.float32)
        mask_tensor = np.expand_dims(mask_tensor, axis=0)

        return image_tensor, mask_tensor

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []

        for yt, yp in zip(y, y_pred_binary):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []

            for yt, yp in zip(y, y_pred_binary):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss /= len(loader)
        epoch_jac /= len(loader)
        epoch_f1 /= len(loader)
        epoch_recall /= len(loader)
        epoch_precision /= len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":
    seeding(42)
    output_files_base_dir = "files_tresunet_segmentation"
    create_dir(output_files_base_dir)
    datetime_object = str(datetime.datetime.now())

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 100
    lr = 1e-4
    early_stopping_patience = 20
    dataset_root_path = r"../dataset/segmentation_dataset"
    mask_types_to_train = {
        'anterior_chamber': 'anterior_chamber',
        'lens': 'lens',
        'nucleus': 'nucleus',
        'right_iris': 'right_iris',
        'left_iris': 'left_iris'
    }
    image_subfolder_name = 'images'

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    for mask_type, mask_subfolder_name in mask_types_to_train.items():
        print(f"\n--- 开始训练 {mask_type} 的分割模型 ---")

        train_log_path = os.path.join(output_files_base_dir, f"train_log_{mask_type}.txt")
        train_log_file = open(train_log_path, "a")
        print_and_save(train_log_file, f"\n--- 训练 {mask_type} 开始 ---")
        print_and_save(train_log_file, f"训练开始时间: {datetime_object}")
        print_and_save(train_log_file, f"训练掩码类型: {mask_type}\n")
        print_and_save(train_log_file, f"图像尺寸: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n")
        print_and_save(train_log_file, f"Early Stopping Patience: {early_stopping_patience}\n")
        print_and_save(train_log_file, "-" * 30 + "\n")

        checkpoint_path = os.path.join(output_files_base_dir, f"checkpoint_{mask_type}.pth")
        train_data_folder = os.path.join(dataset_root_path, 'train')
        val_data_folder = os.path.join(dataset_root_path, 'val')
        train_dataset = OCTDataset(train_data_folder, image_subfolder_name, mask_subfolder_name, size,
                                   transform=transform)
        valid_dataset = OCTDataset(val_data_folder, image_subfolder_name, mask_subfolder_name, size, transform=None)

        if len(train_dataset) == 0:
            print_and_save(train_log_file,
                           f"错误: 训练集目录 '{train_data_folder}' 为空或无法加载数据，跳过 {mask_type} 的训练。")
            train_log_file.close()
            continue
        if len(valid_dataset) == 0:
            print_and_save(train_log_file, f"警告: 验证集目录 '{val_data_folder}' 为空，将无法进行验证和早停。")

        data_str = f"数据集大小:\n训练集: {len(train_dataset)} - 验证集: {len(valid_dataset)}\n"
        print_and_save(train_log_file, data_str)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        model = TResUnet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        loss_fn = DiceBCELoss()
        loss_name = "BCE Dice Loss"
        data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
        print_and_save(train_log_file, data_str)

        """ Training Loop """
        best_valid_metrics = 0.0
        early_stopping_count = 0

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)

            if len(valid_dataset) > 0:
                valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
                scheduler.step(valid_loss)

                if valid_metrics[1] > best_valid_metrics:
                    data_str = f"Epoch {epoch + 1:02}: Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint..."
                    print_and_save(train_log_file, data_str)
                    best_valid_metrics = valid_metrics[1]
                    torch.save(model.state_dict(), checkpoint_path)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    print_and_save(train_log_file,
                                   f"Epoch {epoch + 1:02}: Valid F1 did not improve. Early stopping count: {early_stopping_count}/{early_stopping_patience}")
            else:
                valid_loss, valid_metrics = float('inf'), [0.0] * 4
                scheduler.step(train_loss)
                print_and_save(train_log_file, f"Epoch {epoch + 1:02}: 验证集为空，跳过验证。")

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
            data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
            data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
            print_and_save(train_log_file, data_str)

            if len(valid_dataset) > 0 and early_stopping_count >= early_stopping_patience:
                print_and_save(train_log_file,
                               f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.\n")
                break

        print_and_save(train_log_file, f"--- {mask_type} 模型训练结束 ---")
        train_log_file.close()

    print("\n所有掩码类型的模型训练完成。")
