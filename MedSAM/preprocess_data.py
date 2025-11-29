# preprocess_data.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import transform, io


ORIGINAL_DATA_ROOT = "../dataset/segmentation_dataset"
# 需要先加载数据预处理成npy格式，存储，再使用
PREPROCESSED_DATA_ROOT = "./npy/AS_OCT_Multi_Class"
IMAGE_SIZE = 1024
GT_SIZE = 256

CLASS_MAP = {
    1: ['anterior_chamber'],
    2: ['lens'],
    3: ['left_iris'],
    4: ['right_iris'],
    5: ['nucleus']
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

    save_img_dir = os.path.join(PREPROCESSED_DATA_ROOT, dataset_type, "imgs")
    save_gt_dir = os.path.join(PREPROCESSED_DATA_ROOT, dataset_type, "gts")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_gt_dir, exist_ok=True)

    image_files = sorted(os.listdir(os.path.join(original_dir, "images")))

    for img_name in tqdm(image_files):
        base_name = os.path.splitext(img_name)[0]

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

        gt_256 = np.zeros((GT_SIZE, GT_SIZE), dtype=np.uint8)

        for class_idx, folder_names in CLASS_MAP.items():
            for folder_name in folder_names:
                mask_path = os.path.join(original_dir, folder_name, f"{base_name}.png")
                if os.path.exists(mask_path):
                    mask = io.imread(mask_path, as_gray=True)
                    mask_resized = transform.resize(mask, (GT_SIZE, GT_SIZE), order=0, preserve_range=True,
                                                    anti_aliasing=False).astype(np.uint8)
                    gt_256[mask_resized > 0] = class_idx

        np.save(os.path.join(save_gt_dir, f"{base_name}.npy"), gt_256)


if __name__ == "__main__":
    preprocess_and_save("train")
    preprocess_and_save("val")
    preprocess_and_save("test")
    print("\nPreprocessing complete!")
    print(f"NPY files saved to: {PREPROCESSED_DATA_ROOT}")
