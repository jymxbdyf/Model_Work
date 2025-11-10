"""
数据下载脚本
自动下载Tiny Shakespeare数据集
"""
import os
import urllib.request
import sys
from utils import get_data_dir


def download_tiny_shakespeare(data_dir=None, filename='tiny_shakespeare.txt'):
    """下载Tiny Shakespeare数据集"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    # 如果没有指定data_dir，使用项目根目录的data目录
    if data_dir is None:
        data_dir = get_data_dir()
    
    filepath = os.path.join(data_dir, filename)
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 如果文件已存在，跳过下载
    if os.path.exists(filepath):
        print(f"数据文件已存在: {filepath}")
        return filepath
    
    print(f"正在下载数据集到 {filepath}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"下载完成！")
        return filepath
    except Exception as e:
        print(f"下载失败: {e}")
        print(f"请手动下载数据集:")
        print(f"URL: {url}")
        print(f"保存到: {filepath}")
        return None


if __name__ == "__main__":
    download_tiny_shakespeare()

