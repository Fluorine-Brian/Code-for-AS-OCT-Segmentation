# preprocess_data.py (更新版)

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import transform, io

# --- 1. 设置你的路径 ---
# 原始数据集的根目录
ORIGINAL_DATA_ROOT = "./data"
# 预处理后 .npy 文件保存的根目录
PREPROCESSED_DATA_ROOT = "./npy/AS_OCT_Multi_Class"

# MedSAM期望的输入尺寸
IMAGE_SIZE = 1024
GT_SIZE = 256

# --- 2. 定义类别、文件夹和它们在 .npy 文件中的目标ID ---
CLASS_MAP = {
    1: ['anterior_chamber'],
    2: ['lens'],
    3: ['iris'],  # 假设你已经将左右虹膜合并到了这个文件夹
}


def preprocess_and_save(dataset_type):
    """
    处理指定的数据集类型 ('train', 'val', 'test')
    """
    print(f"--- Processing {dataset_type} dataset ---")

    original_dir = os.path.join(ORIGINAL_DATA_ROOT, dataset_type)
    if not os.path.exists(original_dir):
        print(f"Directory not found: {original_dir}. Skipping.")
        return

    # 创建保存 .npy 文件的目标文件夹
    save_img_dir = os.path.join(PREPROCESSED_DATA_ROOT, dataset_type, "imgs")
    save_gt_dir = os.path.join(PREPROCESSED_DATA_ROOT, dataset_type, "gts")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_gt_dir, exist_ok=True)

    image_files = sorted(os.listdir(os.path.join(original_dir, "images")))

    for img_name in tqdm(image_files):
        base_name = os.path.splitext(img_name)[0]

        # --- 处理图像 (与之前相同) ---
        img_path = os.path.join(original_dir, "images", img_name)
        try:
            image = io.imread(img_path)
        except Exception as e:
            print(f"Could not read image {img_path}: {e}. Skipping.")
            continue

        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        image_1024 = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE), order=3, preserve_range=True,
                                      anti_aliasing=True).astype(np.uint8)
        image_1024_normalized = image_1024 / 255.0
        np.save(os.path.join(save_img_dir, f"{base_name}.npy"), image_1024_normalized)

        # --- 核心修改: 处理并合并多个掩码 ---
        # 创建一个空的掩码 (背景为0)
        gt_256 = np.zeros((GT_SIZE, GT_SIZE), dtype=np.uint8)

        # 遍历我们定义的每个类别
        for class_idx, folder_names in CLASS_MAP.items():
            for folder_name in folder_names:
                # 假设掩码是png格式
                mask_path = os.path.join(original_dir, folder_name, f"{base_name}.png")
                if os.path.exists(mask_path):
                    mask = io.imread(mask_path, as_gray=True)
                    mask_resized = transform.resize(mask, (GT_SIZE, GT_SIZE), order=0, preserve_range=True,
                                                    anti_aliasing=False).astype(np.uint8)
                    # 在合并掩码的对应位置填上类别ID
                    gt_256[mask_resized > 0] = class_idx

        # 保存这个合并后的多标签掩码
        np.save(os.path.join(save_gt_dir, f"{base_name}.npy"), gt_256)


if __name__ == "__main__":
    preprocess_and_save("train")
    preprocess_and_save("val")
    preprocess_and_save("test")
    print("\nPreprocessing complete!")
    print(f"NPY files saved to: {PREPROCESSED_DATA_ROOT}")
