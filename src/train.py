"""
Transformer训练脚本
包含：训练稳定性技巧、消融实验、可视化等功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False
import argparse
import random
import os
import json
from datetime import datetime
from model import Transformer
from utils import get_results_dir, get_data_dir
import sys

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts, vocab, seq_length=50):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        self.vocab_size = len(vocab)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) > self.seq_length + 1:
            start_idx = random.randint(0, len(text) - self.seq_length - 1)
            sequence = text[start_idx:start_idx + self.seq_length + 1]
        else:
            sequence = text + ['<pad>'] * (self.seq_length + 1 - len(text))
        
        input_seq = [self.vocab.get(char, self.vocab['<unk>']) for char in sequence[:-1]]
        target_seq = [self.vocab.get(char, self.vocab['<unk>']) for char in sequence[1:]]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def build_vocab(text):
    """构建词汇表"""
    chars = sorted(list(set(text)))
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    vocab['<bos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    return vocab


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # 生成目标掩码（因果掩码）
        # mask形状: [seq_len, seq_len]，会在模型内部扩展到batch维度
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1) - 1).to(device)
        
        # 前向传播
        output = model(src, tgt[:, :-1], tgt_mask=tgt_mask)
        loss = criterion(output.reshape(-1, model.output_layer.out_features), 
                        tgt[:, 1:].reshape(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # mask形状: [seq_len, seq_len]，会在模型内部扩展到batch维度
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1) - 1).to(device)
            output = model(src, tgt[:, :-1], tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, model.output_layer.out_features), 
                            tgt[:, 1:].reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_model(config):
    """主训练函数"""
    # 设置随机种子
    set_seed(config.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建结果目录
    results_dir = get_results_dir()
    # 如果指定了实验名称，使用实验名称；否则使用时间戳
    if hasattr(config, 'exp_name') and config.exp_name:
        exp_name = config.exp_name
        # 如果文件夹已存在，添加时间戳后缀避免覆盖
        exp_dir = os.path.join(results_dir, exp_name)
        if os.path.exists(exp_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(results_dir, f'{exp_name}_{timestamp}')
            print(f"警告: 实验文件夹已存在，使用新名称: {os.path.basename(exp_dir)}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(results_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_dict = vars(config)
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # 准备数据
    data_path = config.data_path
    # 如果路径是相对路径，转换为相对于项目根目录的路径
    if not os.path.isabs(data_path):
        # 如果路径不包含data目录，则添加到data目录
        if 'data' not in data_path:
            data_dir = get_data_dir()
            filename = os.path.basename(data_path) or 'tiny_shakespeare.txt'
            data_path = os.path.join(data_dir, filename)
        else:
            # 如果已经是data/xxx格式，转换为绝对路径
            data_dir = get_data_dir()
            filename = os.path.basename(data_path)
            data_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在")
        print("尝试自动下载数据集...")
        try:
            from download_data import download_tiny_shakespeare
            data_dir = get_data_dir()
            filename = os.path.basename(data_path) or 'tiny_shakespeare.txt'
            downloaded_path = download_tiny_shakespeare(data_dir, filename)
            if downloaded_path and os.path.exists(downloaded_path):
                data_path = downloaded_path
                print(f"数据集下载成功: {data_path}")
            else:
                print("自动下载失败，请手动下载数据集")
                print(f"URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
                return None, None, None
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载数据集或检查路径")
            return None, None, None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 构建词汇表
    vocab = build_vocab(text)
    vocab_size = len(vocab)
    print(f'词汇表大小: {vocab_size}')
    
    # 保存词汇表
    with open(os.path.join(exp_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    # 分割训练验证集
    chars = list(text)
    split_idx = int(0.9 * len(chars))
    train_chars = chars[:split_idx]
    val_chars = chars[split_idx:]
    
    # 创建数据集（将文本分割成固定长度的序列）
    train_texts = []
    for i in range(0, len(train_chars), config.seq_length * 10):
        train_texts.append(train_chars[i:i + config.seq_length * 10])
    
    val_texts = []
    for i in range(0, len(val_chars), config.seq_length * 10):
        val_texts.append(val_chars[i:i + config.seq_length * 10])
    
    train_dataset = TextDataset(train_texts, vocab, config.seq_length)
    val_dataset = TextDataset(val_texts, vocab, config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f'训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}')
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.d_ff,
        max_seq_length=config.seq_length,
        dropout=config.dropout
    ).to(device)
    
    # 统计参数量
    num_params = model.count_parameters()
    print(f'模型参数量: {num_params:,}')
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # 优化器 (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # 学习率调度器
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.scheduler == 'warmup':
        # 简单的warmup调度器
        warmup_steps = config.warmup_steps
        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return epoch / warmup_steps
            else:
                return max(0.1, (config.epochs - epoch) / (config.epochs - warmup_steps))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # 训练循环
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    
    print(f'\n开始训练，共 {config.epochs} 个epoch...\n')
    
    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}\n')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': config_dict,
                'val_loss': val_loss,
                'num_params': num_params
            }, os.path.join(exp_dir, 'best_model.pth'))
            print(f'  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})\n')
    
    # 保存最终模型
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'config': config_dict,
        'val_loss': val_losses[-1],
        'num_params': num_params
    }, os.path.join(exp_dir, 'final_model.pth'))
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, learning_rates, exp_dir)
    
    # 保存训练结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'num_params': num_params,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\n训练完成！结果保存在: {exp_dir}')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    
    return model, vocab, exp_dir


def plot_training_curves(train_losses, val_losses, learning_rates, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', linewidth=2)
    axes[0].plot(val_losses, label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1].plot(learning_rates, label='学习率', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('学习率调度曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'训练曲线已保存: {os.path.join(save_dir, "training_curves.png")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer训练脚本')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--seq_length', type=int, default=50, help='序列长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 训练稳定性技巧
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'warmup', 'none'], help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR步长')
    parser.add_argument('--gamma', type=float, default=0.8, help='StepLR衰减率')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Warmup步数')
    
    # 其他
    parser.add_argument('--data_path', type=str, default='data/tiny_shakespeare.txt', 
                       help='数据文件路径')
    parser.add_argument('--use_cuda', action='store_true', help='使用GPU')
    parser.add_argument('--exp_name', type=str, default=None, 
                       help='实验名称（如果不指定，将使用时间戳）')
    
    config = parser.parse_args()
    
    # 创建必要的目录
    results_dir = get_results_dir()
    data_dir = get_data_dir()
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    model, vocab, exp_dir = train_model(config)

