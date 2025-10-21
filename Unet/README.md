# UNet医学影像分割训练教程

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn/@ZeyiLin/Unet-Medical-Segmentation/runs/67konj7kdqhnfdmusy2u6/chart)

![](./readme_files/train_image.png)

## 安装环境

```bash
pip install -r requirements.txt
```

## 下载数据集

```bash
python download.py
```

然后你需要解压下载的文件：

```bash
unzip dataset/Brain_Tumor_Image_DataSet.zip -d dataset/
```

## 训练

```bash 
python train.py
```

查看SwanLab训练记录过程：[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn/@ZeyiLin/Unet-Medical-Segmentation/runs/67konj7kdqhnfdmusy2u6/chart)

![](./readme_files/swanlab.png)

## 推理

```bash
python predict.py
```

推理结果会保存在当前目录下，图像名为`predictions.png`。

![](./readme_files/predictions.png)