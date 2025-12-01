# An Annotated AS-OCT Dataset for Multi-Disease Analysis and Ophthalmic Biometry

This project provides the official dataset and benchmark models for our paper, "An Annotated AS-OCT Dataset of Anterior Segment Structures for Multi-Disease Analysis and Ophthalmic Biometry."

Anterior Segment Optical Coherence Tomography (AS-OCT) is essential for modern ophthalmology, providing high-resolution visualization of the anterior eye for diagnosis, surgical planning, and biometric measurement. However, the development of automated analysis tools has been limited by a lack of high-quality, comprehensively annotated public datasets.

To address this gap, this repository presents a curated AS-OCT dataset of 848 images from 488 unique individuals across four clinical categories: normal, cataract, glaucoma, and co-occurrence conditions. Each image underwent strict quality control, and the dataset features comprehensive expert annotations, including segmentation masks for the anterior chamber, iris, lens, and nucleus, as well as localization points for scleral spurs. A key methodological strength is the "one-image-per-patient" design, which ensures subject independence and prevents data leakage, providing a robust foundation for building truly generalizable AI models.

This repository contains the code to reproduce our benchmark results using three state-of-the-art segmentation models (**TRANSRESUNET**, **MEDSAM**, **nnUNetv2**) and one keypoint detection model (**YOLOv11 Pose**).

## Project Workflow

The overall workflow for dataset creation and model validation is illustrated below:

![](flowchart.png)

**(a)** Patient Selection across four categories. **(b)** Image Acquisition using Tomey CASIA2. **(c)** Rigorous Image Quality Assessment and Selection. **(d)** Annotation of key anatomical structures. **(e)** Inter-annotator evaluation to ensure consistency. **(f)** Validation of the dataset using benchmark segmentation and positioning models.

## Table of Contents

- [Project Structure](#project-structure)
- [Model Descriptions](#model-descriptions)
  - [TRANSRESUNET](#transresunet)
  - [MEDSAM](#medsam)
  - [nnUNetv2](#nnunetv2)
  - [YOLOv11 Pose](#yolov11-pose)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation for TransResUNet, MedSAM](#installation-for-transresunet-medsam)
  - [Installation for nnUNetv2 (Separate Environment)](#installation-for-nnunetv2-separate-environment)
  - [Installation for YOLOv11 Pose (Separate Environment)](#installation-for-yolov11-pose-separate-environment)
- [Dataset Preparation](#dataset-preparation)
  - [Unified Raw Data Structure](#unified-raw-data-structure)
  - [Common Segmentation Dataset Preparation (for TransResUNet, MedSAM)](#common-segmentation-dataset-preparation-for-transresunet-medsam)
  - [nnUNetv2 Specific Dataset Preparation](#nnunetv2-specific-dataset-preparation)
  - [YOLOv11 Pose Specific Dataset Preparation](#yolov11-pose-specific-dataset-preparation)
- [Usage](#usage)
  - [1. TRANSRESUNET Workflow](#1-transresunet-workflow)
  - [2. MEDSAM Workflow](#2-medsam-workflow)
  - [3. nnUNetv2 Workflow](#3-nnunetv2-workflow)
  - [4. YOLOv11 Pose Workflow](#4-yolov11-pose-workflow)
- [Evaluation](#evaluation)
- [Results](#results)
- [Attribution and Citation](#attribution-and-citation)
- [License](#license)

## Project Structure

```
./
├── AS-OCT-SS-Detection/       # YOLOv11 Pose detection code and its specific data
│   ├── datasets/              # Output directory for YOLO-converted data
│   ├── convert_asoct_to_yolo_pose.py
│   ├── train_asoct_pose.py
│   ├── ...
├── MedSAM/                    # MedSAM segmentation code
├── TransResUnet/              # TransResUNet segmentation code
├── processing_data.py         # Custom script for TransResUNet/MedSAM data preparation
├── png_to_nii.gz.py           # Custom script for nnUNetv2 data conversion
├── generate_json.py           # Custom script for nnUNetv2 dataset.json generation
├── iou_dice_calculate.py      # Custom script for nnUNetv2 evaluation
├── raw_data/                  # Unified root for all original datasets
│   └── as-oct dataset/        # Contains all raw images and LabelMe JSONs/PNG masks
│       ├── images/
│       ├── lens/
│       ├── left_iris/
│       ├── right_iris/
│       ├── anterior_chamber/
│       ├── nucleus/
│       ├── Cataract/
│       ├── Normal/
│       ├── PACG/
│       └── PACG_Cataract/
├── processed_segmentation_data/ # Output directory for TransResUNet/MedSAM processed data
│   └── segmentation_dataset/
├── OUTPUT_FOLDER/             # nnUNetv2's output directory (nnUNet_raw, etc.)
├── README.md                  # This README file
└── ... (other files like requirements.txt, LICENSE, etc.)
```

## Model Descriptions

### TRANSRESUNET

The **TransResUNet** is an advanced hybrid architecture that enhances the standard U-Net. It integrates residual blocks (from ResNet) to improve feature extraction and prevent vanishing gradients, and incorporates Transformer modules to capture long-range dependencies and global context within the image.

### MEDSAM

The **Medical Segment Anything Model (MEDSAM)** is a specialized version of the Segment Anything Model (SAM) tailored for medical imaging. It is a foundation model pre-trained on a massive dataset of medical images that can be fine-tuned on our AS-OCT data to achieve high-quality segmentation.

### nnUNetv2

The **nnUNetv2** is a self-configuring framework that automatically handles data preprocessing, model architecture selection, and hyperparameter optimization, making it a highly robust benchmark for medical image segmentation.

### YOLOv11 Pose

The **YOLOv11 Pose** model is a state-of-the-art pose estimation model adapted here for keypoint detection. It enables the automatic localization of critical anatomical landmarks like the left and right scleral spurs in AS-OCT images.

## Getting Started

**Note that TransResUNet and MedSAM share one environment, while nnUNetv2 and YOLOv11 Pose each require separate, dedicated environments due to their specific dependencies and Python version requirements.**

### Prerequisites

- Python 3.8+ (for TransResUNet, MedSAM)
- Python 3.10 (for nnUNetv2)
- Python 3.11 (for YOLOv11 Pose)
- Conda or another virtual environment manager
- Git and Git LFS
- CUDA-enabled GPU

### Installation for TransResUNet, MedSAM

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a Conda environment:**

   ```bash
   conda create --name as-oct-seg python=3.9
   conda activate as-oct-seg
   ```

3. **Install the required dependencies:**

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. **Install LabelMe for data annotation and processing:**
   **Note:** The `processing_data.py` script relies on `labelme`. Please ensure it is installed.

   ```bash
   pip install labelme
   ```

### Installation for nnUNetv2 (Separate Environment)

1. **Create and activate a dedicated Conda environment for nnUNetv2:**

   ```bash
   conda create -n nnunetv2_env python=3.10
   conda activate nnunetv2_env
   ```

2. **Install nnUNetv2 and its dependencies:**

   ```bash
   pip install nnunetv2
   pip install nibabel numpy SimpleITK Pillow tqdm
   ```

3. **Configure nnUNetv2 Environment Variables:**
   nnUNetv2 relies on three environment variables. **You must set these to absolute paths on your system.**

   ```bash
   # Replace '/path/to/your/project/root' with the absolute path to this repository
   export nnUNet_raw="/path/to/your/project/root/OUTPUT_FOLDER/nnUNet_raw"
   export nnUNet_preprocessed="/path/to/your/project/root/OUTPUT_FOLDER/nnUNet_preprocessed"
   export nnUNet_results="/path/to/your/project/root/OUTPUT_FOLDER/nnUNet_results"
   
   # Create these directories if they don't exist
   mkdir -p $nnUNet_raw
   mkdir -p $nnUNet_preprocessed
   mkdir -p $nnUNet_results
   ```

### Installation for YOLOv11 Pose (Separate Environment)

1. **Create and activate a dedicated Conda environment for YOLOv11 Pose:**

   ```bash
   conda create -n yolo11 python=3.11 -y
   conda activate yolo11
   ```

2. **Install PyTorch with CUDA support:**

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

3. **Install Ultralytics and other dependencies:**

   ```bash
   cd AS-OCT-SS-Detection/
   pip install -e .
   pip install scikit-learn matplotlib
   cd ..
   ```

## Dataset Preparation

### Unified Raw Data Structure

All raw AS-OCT images and their corresponding annotations should be organized under the `raw_data/as-oct dataset/` directory.

```
./raw_data/as-oct dataset/
├── images/            # Contains all original AS-OCT images (.jpg)
├── lens/              # Contains individual binary PNG masks for 'lens'
├── left_iris/         # ...
├── right_iris/        # ...
├── anterior_chamber/  # ...
├── nucleus/           # ...
├── Cataract/          # Contains original images and LabelMe JSONs (for YOLO)
├── Normal/            # ...
├── PACG/              # ...
└── PACG_Cataract/     # ...
```

### Common Segmentation Dataset Preparation (for TransResUNet, MedSAM)

This process converts raw annotated data into a structured format suitable for training TransResUNet and MedSAM.

**Step 1: Run the Processing Script**
Execute the `processing_data.py` script from the project's root directory. It reads from `raw_data/as-oct dataset/`, generates masks, splits the data (70:15:15), and saves the output to `processed_segmentation_data/segmentation_dataset/`.

```bash
# Ensure you are in the 'as-oct-seg' conda environment
conda activate as-oct-seg
python processing_data.py
```

**Step 2: Final Dataset Structure**
The `processed_segmentation_data/` directory will contain the following structure:

```
./processed_segmentation_data/
└── segmentation_dataset/
    ├── train/
    │   ├── image/
    │   ├── anterior_chamber/
    │   ├── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

### nnUNetv2 Specific Dataset Preparation

This process converts raw data into NIfTI format for nnUNetv2.

**Prerequisite:** Activate the `nnunetv2_env` and set the environment variables.

**Step 1: PNG Images and Masks to NIfTI Format**
Use the `png_to_nii.gz.py` script to convert raw images and masks into `.nii.gz` format and organize them into the `$nnUNet_raw` directory.

```bash
# Ensure you are in the 'nnunetv2_env' conda environment
conda activate nnunetv2_env

TASK_ID="800"
TASK_NAME="AS-OCT"
NNUNET_RAW_TASK_DIR="${nnUNet_raw}/Dataset${TASK_ID}_${TASK_NAME}"

python png_to_nii.gz.py \
    --raw_image_dir "./raw_data/as-oct dataset/images" \
    --lens_mask_dir "./raw_data/as-oct dataset/lens" \
    --left_iris_mask_dir "./raw_data/as-oct dataset/left_iris" \
    --right_iris_mask_dir "./raw_data/as-oct dataset/right_iris" \
    --ac_mask_dir "./raw_data/as-oct dataset/anterior_chamber" \
    --nucleus_mask_dir "./raw_data/as-oct dataset/nucleus" \
    --output_nnunet_raw_dir "${NNUNET_RAW_TASK_DIR}" \
    --case_prefix "asoct" \
    --split_ratio 0.8 \
    --image_extension ".jpg"
```

**Step 2: Generate `dataset.json` File**
Use the `generate_json.py` script to create the `dataset.json` file required by nnUNetv2. **Use the sample counts printed by the previous script.**

```bash
# Ensure you are in the 'nnunetv2_env' conda environment
conda activate nnunetv2_env

TASK_ID="800"
TASK_NAME="AS-OCT"
NNUNET_RAW_TASK_DIR="${nnUNet_raw}/Dataset${TASK_ID}_${TASK_NAME}"
NUM_TRAIN_SAMPLES=675 # Replace with actual count
NUM_TEST_SAMPLES=166  # Replace with actual count

python generate_json.py \
    --task_folder "${NNUNET_RAW_TASK_DIR}" \
    --num_training_samples ${NUM_TRAIN_SAMPLES} \
    --num_test_samples ${NUM_TEST_SAMPLES} \
    --case_prefix "asoct" \
    --task_name "${TASK_NAME}" \
    --labels_json '{"0": "background", "1": "lens", "2": "left_iris", "3": "right_iris", "4": "anterior_chamber", "5": "nucleus"}'
```

**Step 3: nnUNetv2 Data Planning and Preprocessing**
This step is handled automatically by nnUNetv2.

```bash
# Ensure you are in the 'nnunetv2_env' conda environment
conda activate nnunetv2_env
nnUNetv2_plan_and_preprocess -d ${TASK_ID} --verify_dataset_integrity
```

### YOLOv11 Pose Specific Dataset Preparation

This process converts LabelMe JSON files into the YOLO format.

**Step 1: Run the Data Conversion Script**
Execute the `convert_asoct_to_yolo_pose.py` script from the `AS-OCT-SS-Detection/` directory.

```bash
# Ensure you are in the 'yolo11' conda environment
conda activate yolo11
cd AS-OCT-SS-Detection/
python convert_asoct_to_yolo_pose.py
cd ..
```

**Output Location**: The converted data will be saved to `AS-OCT-SS-Detection/datasets/ASOCT_YOLO/`.

**Step 2: Pre-trained Weights (Optional)**
Pre-trained weights will be automatically downloaded during the first training run.

## Usage

### 1. TRANSRESUNET Workflow

1. **Configure Paths:** Open `TransResUnet/train.py` and set dataset paths to `./processed_segmentation_data/segmentation_dataset/`.

2. **Train & Evaluate:**

   ```bash
   conda activate as-oct-seg
   cd TransResUnet/
   python train.py
   python test.py
   cd ..
   ```

### 2. MEDSAM Workflow

1. **Data Pre-processing:**

   ```bash
   conda activate as-oct-seg
   cd MedSAM/
   python preprocess_data.py
   cd ..
   ```

2. **Download & Place Weights:** Download `medsam_vit_b.pth` and place it in `MedSAM/workdir/MedSAM/`.

3. **Fine-tune & Evaluate:**

   ```bash
   conda activate as-oct-seg
   cd MedSAM/
   python train_one_gpu.py
   python evaluate.py
   cd ..
   ```

### 3. nnUNetv2 Workflow

**Prerequisite:** Complete nnUNetv2 installation and data preparation.

1. **Train the Model:**

   ```bash
   conda activate nnunetv2_env
   nnUNetv2_train 800 3d_fullres 0
   ```

2. **Model Inference (Prediction):**

   ```bash
   conda activate nnunetv2_env
   OUTPUT_PREDICTION_DIR="./OUTPUT_FOLDER/nnunetv2_predictions"
   NNUNET_RAW_TASK_DIR="${nnUNet_raw}/Dataset800_AS-OCT"
   mkdir -p ${OUTPUT_PREDICTION_DIR}
   
   nnUNetv2_predict \
       -i "${NNUNET_RAW_TASK_DIR}/imagesTs" \
       -o "${OUTPUT_PREDICTION_DIR}" \
       -d 800 \
       -c 3d_fullres \
       -f all \
       -chk checkpoint_best.pth
   ```

### 4. YOLOv11 Pose Workflow

**Prerequisite:** Complete YOLOv11 Pose installation and data preparation.

1. **Train the Model:**
   Modify `AS-OCT-SS-Detection/train_asoct_pose.py` to select your model, then run:

   ```bash
   conda activate yolo11
   cd AS-OCT-SS-Detection/
   python train_asoct_pose.py
   cd ..
   ```

2. **Evaluate the Model:**

   ```bash
   conda activate yolo11
   cd AS-OCT-SS-Detection/
   python validate_asoct_pose.py
   python evaluate_pixel_distance.py
   python visualize_prediction_errors.py
   cd ..
   ```

3. **Predict with the Model:**

   ```bash
   conda activate yolo11
   cd AS-OCT-SS-Detection/
   python predict_asoct_pose.py
   cd ..
   ```

## Evaluation

For **segmentation models**, performance is measured using **Intersection over Union (IoU)** and **Dice Coefficient**. For **nnUNetv2**, the `iou_dice_calculate.py` script provides detailed metrics including mean, standard deviation, and variance.

For **YOLOv11 Pose detection**, evaluation metrics include **mAP50-95 (pose)**, **Mean Pixel Distance (MPD)**, and **Percentage of Correct Keypoints (PCK)**.

## Results

### Segmentation Results

| Anatomical Structure | Method       | Dice (%)            | IoU (%)             |
| :------------------- | :----------- | :------------------ | :------------------ |
| **Anterior Chamber** | nnUNet       | 0.9878 ± 0.0067     | 0.9760 ± 0.0126     |
|                      | TransResUNet | 0.9764 ± 0.0082     | **0.9880 ± 0.0042** |
|                      | MedSAM       | 0.9598 ± 0.0462     | 0.9788 ± 0.0294     |
| **Lens**             | nnUNet       | 0.9454 ± 0.0071     | 0.9038 ± 0.0088     |
|                      | TransResUNet | **0.9776 ± 0.0078** | **0.9886 ± 0.0040** |
|                      | MedSAM       | 0.6783 ± 0.3032     | 0.7536 ± 0.3052     |
| **Nucleus**          | nnUNet       | 0.9338 ± 0.0099     | 0.8889 ± 0.0200     |
|                      | TransResUNet | **0.9375 ± 0.0325** | **0.9674 ± 0.0178** |
|                      | MedSAM       | 0.8907 ± 0.0567     | 0.9412 ± 0.0336     |
| **Left Iris**        | nnUNet       | 0.7623 ± 0.0935     | 0.6914 ± 0.0952     |
|                      | TransResUNet | 0.8264 ± 0.0547     | 0.9039 ± 0.0341     |
|                      | MedSAM       | **0.8552 ± 0.0437** | **0.9213 ± 0.0262** |
| **Right Iris**       | nnUNet       | **0.8517 ± 0.0230** | 0.7657 ± 0.0345     |
|                      | TransResUNet | 0.7426 ± 0.0611     | 0.8509 ± 0.0418     |
|                      | MedSAM       | 0.8455 ± 0.0420     | **0.9157 ± 0.0253** |

### Scleral Spur Localization Results

| Site  | MPD (px)       | Median (px) | PCK@5px | PCK@10px | PCK@20px | PCK@50px |
| :---- | :------------- | :---------- | :------ | :------- | :------- | :------- |
| Left  | 41.39 ± 23.47  | 39.53       | 1.4%    | 3.5%     | 32.5%    | 69.2%    |
| Right | 34.53 ± 110.50 | 22.98       | 6.2%    | 16.0%    | 41.7%    | 90.3%    |

## Attribution and Citation

This project integrates and builds upon several excellent open-source projects. If you use this code in your research, please cite our paper and the original works:

* **TRANSRESUNET:**

  * GitHub Repository: [https://github.com/nikhilroxtomar/TransResUNet](https://github.com/nikhilroxtomar/TransResUNet)

  * Original Paper:

    ```bibtex
    @misc{tomar2022transresunet,
          title={TransResU-Net: Transformer based ResU-Net for Real-Time Colonoscopy Polyp Segmentation}, 
          author={Nikhil Kumar Tomar and Annie Shergill and Brandon Rieders and Ulas Bagci and Debesh Jha},
          year={2022},
          eprint={2206.08985},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }
    ```

* **MEDSAM:**

  * GitHub Repository: [https://github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)

  * Original Paper:

    ```bibtex
    @article{ma2023medsam,
      title={MedSAM: Segment Anything Model for Medical Images},
      author={Ma, Junyu and Wang, Bo and Li, Shang and Li, Yiming and Li, Cong and Li, Kang and Li, Zongwei and Li, Hong and Li, Xiaomeng and Li, Yuxin and others},
      journal={arXiv preprint arXiv:2304.00716},
      year={2023}
    }
    ```

* **nnUNetv2:**

  * GitHub Repository: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

  * Original Paper:

    ```bibtex
    @article{isensee2021nnu,
      title={nnU-Net: a self-configuring method for deep learning-based medical image segmentation},
      author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
      journal={Nature Methods},
      volume={18},
      number={2},
      pages={203--211},
      year={2021},
      publisher={Nature Publishing Group}
    }
    ```

* **YOLOv11 Pose (Ultralytics):**

  *   GitHub Repository: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
  *   (Please refer to their official documentation for the most up-to-date citation guidelines)

**Our Paper:**
If you use this repository in your work, please also cite our paper:

```bibtex
@article{your_paper_citation,
  title={An Annotated AS-OCT Dataset of Anterior Segment Structures for Multi-Disease Analysis and Ophthalmic Biometry},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={Year}
}
```

## License

This project is licensed under the [LICENSE file in the root directory](LICENSE). Please refer to the `LICENSE` file for full details.