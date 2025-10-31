"""
检查掩码和图像数据数量是否匹配
"""

import os

# 存放 .jpg 图像的文件夹路径
IMAGE_DIR = r"C:\srp_OCT\TransResUNet-main\AS-OCT_Segmentation\Unet\data\train\images"

# 存放 .png 掩码的文件夹路径
MASK_DIR = r"C:\srp_OCT\TransResUNet-main\AS-OCT_Segmentation\Unet\data\train\nucleus"


def check_data_consistency():
    """
    主函数，双向检查图像和掩码文件夹，找出所有不匹配的文件。
    """
    print("--- 开始进行数据集一致性双向检查 ---")
    print(f"图像文件夹: {IMAGE_DIR}")
    print(f"掩码文件夹: {MASK_DIR}\n")

    if not os.path.isdir(IMAGE_DIR):
        print(f"错误: 图像文件夹路径不存在 -> '{IMAGE_DIR}'")
        return
    if not os.path.isdir(MASK_DIR):
        print(f"错误: 掩码文件夹路径不存在 -> '{MASK_DIR}'")
        return
    try:
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')}
        mask_basenames = {os.path.splitext(f)[0] for f in os.listdir(MASK_DIR) if f.lower().endswith('.png')}
    except Exception as e:
        print(f"读取文件列表时出错: {e}")
        return

    print(f"共找到 {len(image_basenames)} 个图像文件。")
    print(f"共找到 {len(mask_basenames)} 个掩码文件。")
    missing_masks = image_basenames - mask_basenames
    orphan_masks = mask_basenames - image_basenames
    print("\n" + "=" * 40)
    print("          一致性检查报告")
    print("=" * 40)

    print("\n[检查 1/2] 缺少对应掩码的图像文件:")
    if not missing_masks:
        print("  - 完美！没有找到缺少掩码的图像文件。")
    else:
        print(f"  - 发现 {len(missing_masks)} 个问题文件:")
        for basename in sorted(list(missing_masks)):
            print(f"    -> {basename}.jpg")

    print("\n[检查 2/2] 没有对应图像的掩码文件 (多余的掩码):")
    if not orphan_masks:
        print("  - 完美！没有找到多余的掩码文件。")
    else:
        print(f"  - 发现 {len(orphan_masks)} 个问题文件:")
        for basename in sorted(list(orphan_masks)):
            print(f"    -> {basename}.png")

    print("\n" + "=" * 40)

    if not missing_masks and not orphan_masks:
        print("\n结论: 恭喜！你的数据集非常干净，所有图像和掩码一一对应。")
    else:
        print("\n结论: 数据集存在不匹配项，请根据上面的报告进行清理。")

    print("\n--- 检查完成 ---")


if __name__ == '__main__':
    check_data_consistency()
