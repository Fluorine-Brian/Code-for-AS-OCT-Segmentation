import requests
import os

def download_file(url, save_path):
    # 发送GET请求下载文件
    print(f"开始下载文件: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查是否下载成功
    
    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))
    
    # 写入文件
    with open(save_path, 'wb') as f:
        if file_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # 显示下载进度
                    progress = int(50 * downloaded / file_size)
                    print(f"\r下载进度: [{'=' * progress}{' ' * (50-progress)}] {downloaded}/{file_size} bytes", end='')
    print("\n下载完成!")

# 设置下载链接和保存路径
url = "https://github.com/Zeyi-Lin/UNet-Medical/releases/download/data/Brain.Tumor.Image.DataSet.zip"
save_path = "dataset/Brain_Tumor_Image_DataSet.zip"

# 创建datasets目录
os.makedirs("dataset", exist_ok=True)

# 执行下载
download_file(url, save_path)
