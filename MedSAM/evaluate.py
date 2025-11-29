# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import glob

join = os.path.join


class MedSAM(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
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


class EvaluationDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "*.npy")))
        self.img_files = [join(self.img_path, os.path.basename(file)) for file in self.gt_path_files]
        print(f"Loaded {len(self.gt_path_files)} data pairs for evaluation from: {data_root}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_1024 = np.load(self.img_files[index])
        gt_multi_label = np.load(self.gt_path_files[index])

        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
        gt_multi_label_tensor = torch.tensor(gt_multi_label).long()

        return img_1024_tensor, gt_multi_label_tensor


def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    pred_mask = (pred_mask > 0.5).astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask | true_mask)

    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (np.sum(pred_mask) + np.sum(true_mask) + smooth)

    return iou, dice


# --- 3. 主评估函数 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str,
        default="npy/AS_OCT_Multi_Class/test",
        help="Path to the npy test data directory."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str,
        default='work_dir/MedSAM-AS-OCT-Finetune-20251022-1945/medsam_model_best.pth',
        help="Path to YOUR FINE-TUNED MedSAM model checkpoint (e.g., work_dir/.../medsam_model_best.pth)."
    )
    parser.add_argument(
        "--original_checkpoint", type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the OFFICIAL MedSAM pretrained checkpoint (medsam_vit_b.pth)."
    )
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    CLASS_NAMES = {
        1: 'anterior_chamber',
        2: 'lens',
        3: 'left_iris',
        4: 'right_iris',
        5: 'nucleus'
    }

    print("Loading model...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.original_checkpoint)

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    print(f"Loading your fine-tuned weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    medsam_model.load_state_dict(checkpoint['model'])

    medsam_model.eval()
    print("Model loaded successfully.")

    test_dataset = EvaluationDataset(args.data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    iou_scores = {id: [] for id in CLASS_NAMES.keys()}
    dice_scores = {id: [] for id in CLASS_NAMES.keys()}

    with torch.no_grad():
        for image, gt_multi_label in tqdm(test_dataloader, desc="Evaluating"):
            image = image.to(device)
            gt_multi_label_np = gt_multi_label.squeeze().numpy()

            present_class_ids = np.unique(gt_multi_label_np)
            present_class_ids = present_class_ids[present_class_ids != 0]

            for class_id in present_class_ids:
                gt_binary_np = (gt_multi_label_np == class_id).astype(np.uint8)
                y_indices, x_indices = np.where(gt_binary_np > 0)

                if len(x_indices) == 0 or np.min(x_indices) == np.max(x_indices) or np.min(y_indices) == np.max(
                        y_indices):
                    continue

                bbox = torch.tensor([[
                    np.min(x_indices), np.min(y_indices),
                    np.max(x_indices), np.max(y_indices)
                ]], device=device)

                pred_1024 = medsam_model(image, bbox)

                pred_256 = F.interpolate(
                    pred_1024,
                    size=(gt_multi_label.shape[1], gt_multi_label.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                iou, dice = calculate_metrics(pred_256, gt_binary_np)
                iou_scores[class_id].append(iou)
                dice_scores[class_id].append(dice)

    print("\n" + "=" * 50)
    print("           MedSAM Fine-tuning Evaluation Report")
    print("=" * 50)
    all_ious, all_dices = [], []
    for class_id, class_name in CLASS_NAMES.items():
        ious, dices = iou_scores[class_id], dice_scores[class_id]
        if not ious:
            print(f"\nClass: {class_name} (ID: {class_id})\n  No samples found in the test set.")
            continue
        mean_iou, std_iou = np.mean(ious), np.std(ious)
        mean_dice, std_dice = np.mean(dices), np.std(dices)
        all_ious.extend(ious)
        all_dices.extend(dices)
        print(f"\nClass: {class_name} (ID: {class_id})")
        print(f"  Samples: {len(ious)}")
        print(f"  Mean IoU:   {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"  Mean Dice:  {mean_dice:.4f} ± {std_dice:.4f}")
    print("-" * 50)
    if all_ious:
        print("Overall Performance:")
        print(f"  Mean IoU over all classes:  {np.mean(all_ious):.4f}")
        print(f"  Mean Dice over all classes: {np.mean(all_dices):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
