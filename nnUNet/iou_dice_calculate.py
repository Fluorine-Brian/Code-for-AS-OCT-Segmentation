import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import OrderedDict
from multiprocessing import Pool
from typing import Tuple, Dict, List, Optional

# ----------------------------------------------------
# 1. 配置路径 (根据您的实际路径设置)
# ----------------------------------------------------
# 预测结果路径 (来自 nnUNetv2_predict 的输出)
PREDICTIONS_FOLDER = r"/home/fxy/nnUNet/nnUNetFrame/DATASET/nnUNet_results/test_results/Dataset800_AS-OCT"
# Ground Truth 标签路径 (来自您刚才生成的 labelsTs)
LABELS_GT_FOLDER = r"/home/fxy/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset800_AS-OCT/labelsTs"

# ----------------------------------------------------
# 2. 评估参数
# ----------------------------------------------------
# 类别名称与标签 ID (1, 2, 3, 4, 5) 对应
CLASSES_TO_EVALUATE = {
    1: "lens",
    2: "left_iris",
    3: "right_iris",
    4: "anterior_chamber",
    5: "nucleus"
}
CLASSES_TO_EVALUATE_REVERSE = {v: k for k, v in CLASSES_TO_EVALUATE.items()}

# ----------------------------------------------------
# 3. 核心指标计算函数 (Dice 和 Jaccard/IoU)
# ----------------------------------------------------

def compute_metric(
    prediction_file: str, 
    gt_folder: str, 
    class_id: int, 
    class_name: str
) -> Optional[Tuple[str, int, float, float]]:
    """计算单个文件、单个类别的 Dice 和 Jaccard (IoU)"""
    try:
        # 1. 读取预测结果
        pred_path = os.path.join(PREDICTIONS_FOLDER, prediction_file)
        pred_itk = sitk.ReadImage(pred_path)
        pred_np = sitk.GetArrayFromImage(pred_itk).astype(np.uint8)

        # 2. 构造 GT 路径
        case_id = prediction_file.replace('.nii.gz', '')
        gt_path = os.path.join(gt_folder, f"{case_id}.nii.gz")
        
        if not os.path.exists(gt_path):
             # print(f"Warning: GT file not found for {case_id} at {gt_path}. Skipping.")
             return None

        # 3. 读取 Ground Truth
        gt_itk = sitk.ReadImage(gt_path)
        gt_np = sitk.GetArrayFromImage(gt_itk).astype(np.uint8)
        
        # 确保 GT 和 Prediction 维度匹配
        if gt_np.shape != pred_np.shape:
            # print(f"Warning: Shape mismatch for {case_id}. Skipping.")
            return None

        # 4. 提取当前类别的二值掩码
        pred_binary = (pred_np == class_id)
        gt_binary = (gt_np == class_id)
        
        # 如果 GT 或 Pred 中当前类别完全不存在，则跳过
        if not gt_binary.any() and not pred_binary.any():
             # 如果两者都缺失，Dice/IoU 定义为 1.0 (完美匹配)
             return case_id, class_id, 1.0, 1.0 
        
        # 5. 计算交集和并集
        intersection = np.sum(pred_binary & gt_binary)
        union = np.sum(pred_binary | gt_binary)
        
        # 6. 计算 Dice 和 Jaccard (IoU)
        dice_score = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary))
        jaccard_score = intersection / union if union > 0 else 0.0

        return case_id, class_id, dice_score, jaccard_score

    except Exception as e:
        print(f"Error processing {prediction_file} for class {class_name}: {e}")
        return None


def run_evaluation():
    """主评估逻辑，收集并汇总结果"""
    print(f"--- 🚀 开始评估 nnU-Netv2 结果 (含方差和标准差) ---")

    # 获取所有预测文件 (.nii.gz 文件)
    prediction_files = [f for f in os.listdir(PREDICTIONS_FOLDER) if f.endswith('.nii.gz')]
    if not prediction_files:
        print("错误：预测文件夹中未找到任何 .nii.gz 文件。请检查路径或预测是否成功。")
        return

    print(f"找到 {len(prediction_files)} 个预测文件。")

    # 构造多进程任务列表
    tasks = []
    for filename in prediction_files:
        for class_id, class_name in CLASSES_TO_EVALUATE.items():
            tasks.append((filename, LABELS_GT_FOLDER, class_id, class_name))

    # 使用多进程加速计算
    num_processes = os.cpu_count() or 4
    results = []
    with Pool(num_processes) as p:
        results = list(tqdm(p.starmap(compute_metric, tasks), total=len(tasks), desc="计算指标"))
    
    # 过滤掉 None 的结果
    results = [res for res in results if res is not None]

    # ----------------------------------------------------
    # 4. 结果汇总与展示 (新增方差和标准差)
    # ----------------------------------------------------
    
    # 存储每个类别的 Dice 和 IoU 列表
    metrics_by_class: Dict[int, Dict[str, List[float]]] = {
        cid: {"Dice": [], "IoU": []} for cid in CLASSES_TO_EVALUATE.keys()
    }
    
    for case_id, class_id, dice, jaccard in results:
        metrics_by_class[class_id]["Dice"].append(dice)
        metrics_by_class[class_id]["IoU"].append(jaccard)

    # 计算平均值、方差和标准差
    final_metrics = OrderedDict()
    
    for class_id, class_name in CLASSES_TO_EVALUATE.items():
        dice_scores = metrics_by_class[class_id]["Dice"]
        iou_scores = metrics_by_class[class_id]["IoU"]
        
        if dice_scores:
            final_metrics[class_name] = {
                "Mean Dice": np.mean(dice_scores),
                "Variance Dice": np.var(dice_scores),
                "Std Dice": np.std(dice_scores),
                "Mean IoU": np.mean(iou_scores),
                "Variance IoU": np.var(iou_scores),
                "Std IoU": np.std(iou_scores),
                "Cases": len(dice_scores)
            }

    # 打印结果
    print("\n--- 📊 最终评估结果 (Mean, Std, Variance) ---")
    
    # 打印表头
    header = f"{'Class':<20} | {'Mean Dice':<10} | {'Std Dice':<10} | {'Var Dice':<10} | {'Mean IoU':<10} | {'Std IoU':<10} | {'Var IoU':<10} | {'Cases':<5}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    all_dice_scores = []
    
    for class_name, metrics in final_metrics.items():
        print(
            f"{class_name:<20} | "
            f"{metrics['Mean Dice']:.4f}  | "
            f"{metrics['Std Dice']:.4f}  | "
            f"{metrics['Variance Dice']:.4f}  | "
            f"{metrics['Mean IoU']:.4f}  | "
            f"{metrics['Std IoU']:.4f}  | "
            f"{metrics['Variance IoU']:.4f}  | "
            f"{metrics['Cases']:<5}"
        )
        all_dice_scores.append(metrics['Mean Dice'])
        
    # 计算平均 Dice (Mean Dice over all classes)
    if all_dice_scores:
        mean_dice_overall = np.mean(all_dice_scores)
        print("-" * len(header))
        print(f"{'Overall Mean Dice':<20} | {mean_dice_overall:.4f}")
    
    # 将结果保存为 JSON 文件
    output_json_path = os.path.join(PREDICTIONS_FOLDER, "evaluation_metrics_with_variance.json")
    with open(output_json_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"\n详细结果已保存至: {output_json_path}")
    
if __name__ == "__main__":
    # 确保 SimpleITK 和 numpy 在多进程中能正确运行
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    run_evaluation()