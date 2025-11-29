import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# --- 1. 输入路径 ---
raw_image_dir = r"/home/fxy/nnunet_dataset/images"
lens_mask_dir = r"/home/fxy/nnunet_dataset/lens"
left_iris_mask_dir = r"/home/fxy/nnunet_dataset/left_iris"
right_iris_mask_dir = r"/home/fxy/nnunet_dataset/right_iris"
ac_mask_dir = r"/home/fxy/nnunet_dataset/anterior_chamber"
nucleus_mask_dir = r"/home/fxy/nnunet_dataset/nucleus"

# --- 2. 【核心路径】nnUNet 原始数据路径 (Task800_AS-OCT) ---
# 确保这个路径结构符合 nnUNet v2 的预期
output_base_dir = r"/home/fxy/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset800_AS-OCT" 
# 注意: 为了与您之前成功的预测命令 (Dataset800_AS-OCT) 保持一致，我将路径末尾修正为 Dataset800_AS-OCT
# nnUNetv2_predict -d 800 对应 Dataset800_AS-OCT

# --- 3. 定义输出子文件夹路径 ---
output_imagesTr_dir = os.path.join(output_base_dir, 'imagesTr')
output_labelsTr_dir = os.path.join(output_base_dir, 'labelsTr')
output_imagesTs_dir = os.path.join(output_base_dir, 'imagesTs')
# --- 【新增】测试集标签路径 ---
output_labelsTs_dir = os.path.join(output_base_dir, 'labelsTs')


# --- 4. 确认并创建所有输出目录 ---
print(f"正在创建或确认输出目录: {output_base_dir}...")
os.makedirs(output_imagesTr_dir, exist_ok=True)
os.makedirs(output_labelsTr_dir, exist_ok=True)
os.makedirs(output_imagesTs_dir, exist_ok=True)
os.makedirs(output_labelsTs_dir, exist_ok=True) # 确保 labelsTs 存在
print(f"输出目录已就绪.")

# --- 5. 标签映射 ---
label_map = {
    "lens": 1,
    "left_iris": 2,
    "right_iris": 3,
    "anterior_chamber": 4,
    "nucleus": 5
}
# 简化掩码文件夹列表，便于循环
mask_dirs = {
    "lens": lens_mask_dir,
    "left_iris": left_iris_mask_dir,
    "right_iris": right_iris_mask_dir,
    "anterior_chamber": ac_mask_dir,
    "nucleus": nucleus_mask_dir
}

# --- 6. 文件处理 ---
image_files = sorted([f for f in os.listdir(raw_image_dir) if f.endswith('.jpg')])

train_files = image_files[:675]
test_files = image_files[675:]

print(f"找到 {len(image_files)} 张图像，其中 {len(train_files)} 用于训练，{len(test_files)} 用于测试。")

# ----------------------------------------------------
# A. 处理训练集 (imagesTr 和 labelsTr)
# ----------------------------------------------------
# print("正在处理训练集 (imagesTr 和 labelsTr)...")
# for i, filename in enumerate(tqdm(train_files)):
#     case_identifier = f"asoct_{i:03d}"

#     img_path = os.path.join(raw_image_dir, filename)
#     img_pil = Image.open(img_path).convert('L')
#     img_np = np.array(img_pil)
#     # 转换为 nnU-Net 2D 格式 (1, H, W)
#     img_np = np.expand_dims(img_np, axis=0)
    
#     # --- 图像 Sitk 转换和保存到 imagesTr ---
#     img_sitk = sitk.GetImageFromArray(img_np)
#     sitk.WriteImage(img_sitk, os.path.join(output_imagesTr_dir, f"{case_identifier}_0000.nii.gz"))

#     # --- 标签合并逻辑 ---
#     height, width = img_np.shape[1], img_np.shape[2]
#     merged_mask_np = np.zeros((height, width), dtype=np.uint8)
#     base_filename = os.path.splitext(filename)[0]

#     # 依次合并掩码
#     for mask_name, mask_dir in mask_dirs.items():
#         mask_path = os.path.join(mask_dir, base_filename + '.png')
#         if os.path.exists(mask_path):
#              # 确保 PNG 掩码是单通道或正确处理
#             mask_data = np.array(Image.open(mask_path).convert('L'))
#             merged_mask_np[mask_data > 0] = label_map[mask_name]
    
#     # --- 标签 Sitk 转换和保存到 labelsTr ---
#     merged_mask_np_3d = np.expand_dims(merged_mask_np, axis=0)
#     merged_mask_sitk = sitk.GetImageFromArray(merged_mask_np_3d)

#     # 复制几何信息 (关键步骤)
#     merged_mask_sitk.CopyInformation(img_sitk)

#     # 【修复命名】nnU-Net 的 Ground Truth 标签文件命名格式是 CASE_IDENTIFIER.nii.gz
#     sitk.WriteImage(merged_mask_sitk, os.path.join(output_labelsTr_dir, f"{case_identifier}.nii.gz"))


# ----------------------------------------------------
# B. 处理测试集 (imagesTs 和 labelsTs) - 【重点】
# ----------------------------------------------------
print("正在处理测试集 (imagesTs 和 labelsTs)...")
for i, filename in enumerate(tqdm(test_files)):
    # 确保 case_identifier 连续且唯一
    case_identifier = f"asoct_{len(train_files) + i:03d}"

    img_path = os.path.join(raw_image_dir, filename)
    img_pil = Image.open(img_path).convert('L')
    img_np = np.array(img_pil)
    img_np = np.expand_dims(img_np, axis=0)

    # --- 图像 Sitk 转换和保存到 imagesTs ---
    img_sitk = sitk.GetImageFromArray(img_np)
    sitk.WriteImage(img_sitk, os.path.join(output_imagesTs_dir, f"{case_identifier}_0000.nii.gz"))
    # imagesTs 文件命名格式: CASE_IDENTIFIER_0000.nii.gz

    # --- 【新增】标签合并和保存逻辑 for labelsTs ---
    height, width = img_np.shape[1], img_np.shape[2]
    merged_mask_np = np.zeros((height, width), dtype=np.uint8)
    base_filename = os.path.splitext(filename)[0]

    # 依次合并掩码
    for mask_name, mask_dir in mask_dirs.items():
        mask_path = os.path.join(mask_dir, base_filename + '.png')
        if os.path.exists(mask_path):
            mask_data = np.array(Image.open(mask_path).convert('L'))
            merged_mask_np[mask_data > 0] = label_map[mask_name]

    # 标签 Sitk 转换
    merged_mask_np_3d = np.expand_dims(merged_mask_np, axis=0)
    merged_mask_sitk = sitk.GetImageFromArray(merged_mask_np_3d)

    # 复制几何信息 (关键步骤)
    merged_mask_sitk.CopyInformation(img_sitk)

    # 【重点】保存标签文件到 labelsTs
    # labelsTs 文件命名格式: CASE_IDENTIFIER.nii.gz (与 imagesTs 对应)
    sitk.WriteImage(merged_mask_sitk, os.path.join(output_labelsTs_dir, f"{case_identifier}_0000.nii.gz"))

print("数据转换完成！所有训练集和测试集数据（包括 labelsTs）已生成。")