# AS-OCT Semantic Segmentation Project

This project provides a framework for semantic segmentation of Anterior Segment Optical Coherence Tomography (AS-OCT) images. It implements and compares three different deep learning models: **UNET**, **TRANSRESUNET**, and **MEDSAM** for segmenting four key anatomical structures: the anterior chamber, lens, iris, and nucleus.

## Table of Contents
- [Introduction](#introduction)
- [Model Descriptions](#model-descriptions)
  - [UNET](#unet)
  - [TRANSRESUNET](#transresunet)
  - [MEDSAM](#medsam)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [1. UNET Workflow](#1-unet-workflow)
  - [2. TRANSRESUNET Workflow](#2-transresunet-workflow)
  - [3. MEDSAM Workflow](#3-medsam-workflow)
- [Evaluation](#evaluation)
- [Results](#results)


## Introduction

Semantic segmentation of AS-OCT images is crucial for the quantitative analysis of anterior segment structures, aiding in the diagnosis and management of ocular diseases. This repository offers a comprehensive toolkit to train and evaluate state-of-the-art models for this task. The goal is to accurately delineate the boundaries of the anterior chamber, lens, iris, and nucleus from grayscale AS-OCT scans.

## Model Descriptions

### UNET
The **U-Net** is a convolutional neural network (CNN) architecture designed specifically for biomedical image segmentation. Its characteristic U-shaped structure consists of an encoder path to capture context and a symmetric decoder path for precise localization. Skip connections between the encoder and decoder paths allow the network to combine deep, semantic features with shallow, high-resolution features, leading to excellent performance in segmentation tasks.

### TRANSRESUNET
The **TransResUNet** is an advanced hybrid architecture that enhances the standard U-Net. It integrates residual blocks (from ResNet) to improve feature extraction and prevent vanishing gradients, and incorporates Transformer modules to capture long-range dependencies and global context within the image. This combination makes it particularly powerful for complex segmentation tasks where understanding the relationships between distant pixels is important.

### MEDSAM
The **Medical Segment Anything Model (MEDSAM)** is a specialized version of the Segment Anything Model (SAM) tailored for medical imaging. It is a foundation model pre-trained on a massive dataset of medical images. Instead of training from scratch, MEDSAM can be fine-tuned on a smaller, specific dataset (like our AS-OCT data) to achieve high-quality segmentation with relatively little training data. This approach is known as transfer learning.

## Getting Started

Follow these instructions to set up the project environment and prepare the data.

### Prerequisites
- Python 3.8+
- Conda or another virtual environment manager
- Git and Git LFS (for handling large model files)

### Installation

1.  **Clone the repository:**
    ```bash
    $ git clone https://github.com/Fluorine-Brian/Code-for-AS-OCT-Segmentation.git
    $ cd Code-for-AS-OCT-Segmentation
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    $ conda create --name as-oct-seg python=3.9
    $ conda activate as-oct-seg
    ```

3.  **Install the required dependencies:**
    ```bash
    $ pip install -r requirements.txt
    ```

## Dataset Preparation

All three models expect the dataset to be organized in a specific structure. Please create a `data` directory (or similar) and arrange your files as follows.

The dataset should be split into `train`, `val` (validation), and `test` sets. Each set must contain the original grayscale images and the corresponding PNG masks for each of the four classes.

**Expected Directory Structure:**

```
./data/
├── train/
│   ├── image/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── anterior_chamber_mask/
│   │   ├── 001.png
│   │   └── ...
│   ├── lens_mask/
│   │   ├── 001.png
│   │   └── ...
│   ├── iris_mask/
│   │   ├── 001.png
│   │   └── ...
│   └── nucleus_mask/
│       ├── 001.png
│       └── ...
├── val/
│   ├── image/
│   ├── anterior_chamber_mask/
│   ├── lens_mask/
│   ├── iris_mask/
│   └── nucleus_mask/
└── test/
    ├── image/
    ├── anterior_chamber_mask/
    ├── lens_mask/
    ├── iris_mask/
    └── nucleus_mask/
```

## Usage

Instructions for training and evaluating each model are provided below.

### 1. UNET Workflow

The UNET model code is located in the `Unet/` directory.

1.  **Configure Paths:**
    Open the `Unet/train.py` file and modify the dataset path variables to point to your `train` and `val` directories.

2.  **Train the Model:**
    Navigate to the UNET directory and run the training script.
    ```bash
    $ cd Unet/
    $ python train.py
    ```
    The trained model weights (e.g., `.pth` files) will be saved in the same directory.

3.  **Predict and Evaluate:**
    Once training is complete, run the prediction script to evaluate the model on the test set.
    ```bash
    $ python predict.py
    ```
    This script will generate segmentation masks for the test images and output the Intersection over Union (IoU) and Dice coefficient scores for each class.

### 2. TRANSRESUNET Workflow

The TransResUNet model code is located in the `TransResUnet/` directory (assuming this is the name).

1.  **Configure Paths:**
    Open the `TransResUnet/train.py` file and set the dataset paths correctly.

2.  **Train the Model:**
    ```bash
    $ cd TransResUnet/
    $ python train.py
    ```

3.  **Evaluate the Model:**
    Run the test script to evaluate performance on the test set.
    ```bash
    $ python test.py  # Or the relevant evaluation script
    ```
    This will output the segmentation masks, IoU, and Dice scores.

### 3. MEDSAM Workflow

The MEDSAM workflow involves a pre-processing step and requires a pre-trained model for fine-tuning.

1.  **Data Pre-processing:**
    The dataset (structured as described above) first needs to be converted into the `.npy` format required by the model.
    ```bash
    $ cd MedSAM/
    $ python preprocess_data.py
    ```
    Ensure the paths inside this script are correctly set to your dataset location.

2.  **Download Pre-trained Weights:**
    Download the official MEDSAM pre-trained model weights. You can typically find the link in the original MEDSAM repository. Let's assume the file is `medsam_vit_b.pth`.

3.  **Place Weights:**
    Create the necessary directory and place the downloaded weights inside it:
    ```bash
    $ mkdir -p workdir/MedSAM
    $ mv /path/to/downloaded/medsam_vit_b.pth workdir/MedSAM/
    ```

4.  **Fine-tune the Model:**
    Now, fine-tune the model on your pre-processed AS-OCT data.
    ```bash
    $ python train_one_gpu.py
    ```

5.  **Evaluate the Model:**
    After fine-tuning, run the evaluation script.
    ```bash
    $ python evaluate.py
    ```
    This will generate segmentation results and calculate the IoU and Dice metrics.

## Evaluation

The performance of all models is measured using two standard segmentation metrics:
-   **Intersection over Union (IoU):** Also known as the Jaccard index, it measures the overlap between the predicted mask and the ground truth mask.
-   **Dice Coefficient:** Also known as the F1 score, it is another widely used metric to gauge the similarity between the prediction and the ground truth.

The evaluation scripts for each model will print these scores to the console.

## Results

*(This is a placeholder section. You can add your final results here, such as a comparison table of IoU/Dice scores for the different models and classes, or showcase some of the best segmentation examples.)*

| Model            | Class            | Mean IoU | Mean Dice |
| ---------------- | ---------------- | -------- | --------- |
| **UNET**         | Anterior Chamber | -        | -         |
|                  | Lens             | -        | -         |
|                  | ...              | -        | -         |
| **TRANSRESUNET** | Anterior Chamber | -        | -         |
|                  | ...              | -        | -         |
| **MEDSAM**       | Anterior Chamber | -        | -         |
|                  | ...              | -        | -         |

---