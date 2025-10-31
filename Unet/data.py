# data.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiClassSegmentationDataset(Dataset):
    def __init__(self, root_dir, classes, image_transform=None, mask_transform=None):
        """
        Args:
            root_dir (string): 数据集目录 (e.g., './dataset/train/')
            classes (list): 包含所有类别名称的列表.
            image_transform (callable, optional): 应用于图像的变换.
            mask_transform (callable, optional): 应用于掩码的变换.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.classes = classes
        self.mask_dirs = [os.path.join(root_dir, cls_name) for cls_name in self.classes]

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")

        masks = []
        base_name = os.path.splitext(img_name)[0]

        for mask_dir in self.mask_dirs:
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            mask = Image.open(mask_path).convert("L")
            masks.append(mask)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            processed_masks = [self.mask_transform(m) for m in masks]
            all_masks = torch.cat(processed_masks, dim=0)
            all_masks = (all_masks > 0.5).float()

        return image, all_masks
