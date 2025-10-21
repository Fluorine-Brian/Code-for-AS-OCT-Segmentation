# train_medsam.py (完整最终版)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


# --- 数据集加载类 (已为你的AS-OCT数据定制) ---
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        self.img_files = [join(self.img_path, os.path.basename(file)) for file in self.gt_path_files]
        self.bbox_shift = bbox_shift
        print(f"Loaded {len(self.gt_path_files)} data pairs from: {data_root}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # 加载预处理好的图像和掩码
        img_1024 = np.load(self.img_files[index])
        gt = np.load(self.gt_path_files[index])

        # 从多标签掩码中随机选择一个非背景类别
        label_ids = np.unique(gt)
        label_ids = label_ids[label_ids != 0]  # 排除背景类别0

        if len(label_ids) == 0:
            # 如果这个样本没有任何前景掩码，就随机取下一个
            return self.__getitem__((index + 1) % len(self))

        target_id = random.choice(label_ids)
        # 为选中的类别创建二值掩码 (前景为1, 背景为0)
        gt2D = (gt == target_id).astype(np.uint8)

        # 从二值掩码计算边界框
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 对边界框进行随机扰动，增加数据多样性
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        # 将图像和掩码转换为Tensor
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
        gt2D_tensor = torch.tensor(gt2D[None, :, :]).float()

        return (
            img_1024_tensor,
            gt2D_tensor,
            torch.tensor(bboxes).float(),
        )


# --- 命令行参数设置 (已简化并适配你的任务) ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--tr_npy_path", type=str,
    default="npy/AS_OCT_Multi_Class/train",  # <-- 默认指向你的训练数据
    help="Path to the training npy files directory which contains 'imgs' and 'gts' subfolders.",
)
parser.add_argument("-task_name", type=str, default="MedSAM-AS-OCT-Finetune")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",  # <-- 必须指向你下载的官方MedSAM权重
    help="The path to the official MedSAM pretrained checkpoint."
)
parser.add_argument("-work_dir", type=str, default="./work_dir")
# 训练超参数
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# 优化器参数
parser.add_argument("-weight_decay", type=float, default=0.01)
parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate")
# 其他
parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume training from.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("-use_amp", action="store_true", default=False, help="Use mixed precision training.")
args = parser.parse_args()


# --- 模型定义 (与原脚本相同) ---
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # 冻结提示编码器，因为我们不训练它
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=box_torch, masks=None)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


# --- 主训练函数 ---
def main():
    # 设置保存路径
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    # 加载MedSAM模型
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # 实现正确的微调策略: 冻结图像编码器
    print("Freezing the image encoder...")
    for name, param in medsam_model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = False

    medsam_model.train()

    # 设置优化器，只优化掩码解码器的参数
    optimizer = torch.optim.AdamW(medsam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 设置损失函数
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # 准备数据集
    train_dataset = NpyDataset(args.tr_npy_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 开始训练循环
    losses = []
    best_loss = 1e10
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
            image, gt2D = image.to(device), gt2D.to(device)
            boxes = boxes.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # 1. 获取模型的高分辨率预测 (1024x1024)
                    medsam_pred_1024 = medsam_model(image, boxes)

                    # --- 核心修复: 将预测下采样到与GT相同的尺寸 (256x256) ---
                    medsam_pred_256 = F.interpolate(
                        medsam_pred_1024,
                        size=(gt2D.shape[2], gt2D.shape[3]),  # 动态获取GT的尺寸
                        mode="bilinear",
                        align_corners=False,
                    )

                    # 2. 使用下采样后的预测计算损失
                    loss = seg_loss(medsam_pred_256, gt2D) + ce_loss(medsam_pred_256, gt2D)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # 1. 获取模型的高分辨率预测 (1024x1024)
                medsam_pred_1024 = medsam_model(image, boxes)

                # --- 核心修复: 将预测下采样到与GT相同的尺寸 (256x256) ---
                medsam_pred_256 = F.interpolate(
                    medsam_pred_1024,
                    size=(gt2D.shape[2], gt2D.shape[3]),  # 动态获取GT的尺寸
                    mode="bilinear",
                    align_corners=False,
                )

                # 2. 使用下采样后的预测计算损失
                loss = seg_loss(medsam_pred_256, gt2D) + ce_loss(medsam_pred_256, gt2D)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)

        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}')

        # 保存最新的模型
        checkpoint = {"model": medsam_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

        # 如果当前损失是最好的，则保存为最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print(f"  -> New best model saved with loss: {best_loss:.4f}")

        # 绘制并保存损失曲线图
        plt.plot(losses)
        plt.title("Dice + BCE Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, "train_loss_curve.png"))
        plt.close()


if __name__ == "__main__":
    main()
