import json
import os

# --- 请根据你的情况修改这里的配置 ---
# 你的任务文件夹的完整路径
task_folder = r"/home/fxy/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task800_AS-OCT"
# 你的训练样本数量
num_training_samples = 675
# 你的 case_identifier 前缀 (与你之前生成数据时使用的前缀一致)
case_prefix = 'asoct'
# ------------------------------------

# 定义 dataset.json 的基本结构
dataset_info = {
    "name": "AS-OCT",
    "description": "Anterior Segment Optical Coherence Tomography",
    "reference": "Your Name or Institution",
    "licence": "Private",
    "release": "1.0 05/11/2025",
    "tensorImageSize": "3D",  # 即使是2D图像，v1也通常标记为3D
    "modality": {
        "0": "OCT"
    },
    "labels": {
        "0": "background",
        "1": "lens",
        "2": "left_iris",
        "3": "right_iris",
        "4": "anterior_chamber",
        "5": "nucleus"
    },
    "numTraining": num_training_samples,
    "training": [],  # 我们将用代码填充这个列表
    "test": []  # 如果有测试集，也可以用类似方法填充
}

# --- 自动生成 training 列表 ---
training_list = []
for i in range(num_training_samples):
    case_identifier = f"{case_prefix}_{i:03d}" # e.g., asoct_000, asoct_001, ...
    image_path = f"./imagesTr/{case_identifier}_0000.nii.gz"
    label_path = f"./labelsTr/{case_identifier}_0000.nii.gz"
    training_list.append({
        "image": image_path,
        "label": label_path
    })

dataset_info["training"] = training_list

# --- 自动生成 test 列表 (可选) ---
# 假设你有20个测试样本，从第80号开始
num_test_samples = 166
test_list = []
for i in range(num_test_samples):
    # case_identifier 从训练集最后一个继续
    case_identifier = f"{case_prefix}_{num_training_samples + i:03d}" # e.g., asoct_080, ...
    image_path = f"./imagesTs/{case_identifier}_0000.nii.gz"
    test_list.append(image_path)

dataset_info["test"] = test_list


# --- 将最终的字典写入 dataset.json 文件 ---
output_json_path = os.path.join(task_folder, 'dataset.json')
with open(output_json_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)

print(f"成功生成 dataset.json 文件，已保存至: {output_json_path}")
print(f"文件中包含 {len(dataset_info['training'])} 个训练样本。")

