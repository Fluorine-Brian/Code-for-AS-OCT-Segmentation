# data.py (完整替换)

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BinarySegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        """
        Args:
            root_dir (string): 数据集目录 (e.g., './AS_OCT_Anterior_Chamber/train/')
            image_transform (callable, optional): 应用于图像的变换.
            mask_transform (callable, optional): 应用于掩码的变换.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # 获取所有图像文件名，并排序以确保图像和掩码对应
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        # 确保图像和掩码数量一致
        assert len(self.image_files) == len(self.mask_files), "图像和掩码的数量不匹配"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 构建文件路径
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 加载图像和掩码
        # 图像是灰度的，所以用 "L" 模式
        image = Image.open(img_path).convert("L")
        # 掩码也是灰度的
        mask = Image.open(mask_path).convert("L")

        # 应用变换
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            # 将掩码的值从 [0, 1] 转换为严格的 0 或 1 (二值化)
            mask = (mask > 0.5).float()

        return image, mask
