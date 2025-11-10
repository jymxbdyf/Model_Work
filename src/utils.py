"""
工具函数
"""
import torch
import numpy as np
import random
import os


def get_project_root():
    """获取项目根目录路径"""
    # 获取当前文件的目录（src/）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 返回上一级目录（项目根目录）
    return os.path.dirname(current_dir)


def get_results_dir():
    """获取results目录路径"""
    return os.path.join(get_project_root(), 'results')


def get_data_dir():
    """获取data目录路径"""
    return os.path.join(get_project_root(), 'data')


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, optimizer, epoch, loss, vocab, config, filepath):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab': vocab,
        'config': config
    }, filepath)


def load_model(filepath, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

