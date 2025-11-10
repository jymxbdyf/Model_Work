"""
消融实验脚本
测试不同配置对模型性能的影响
"""
import torch
import argparse
import json
import os
import sys
from datetime import datetime
from train import train_model, set_seed
from download_data import download_tiny_shakespeare
from utils import get_results_dir, get_data_dir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def run_ablation_study(base_config, ablation_configs):
    """运行消融实验"""
    results = []
    
    for i, (name, config_updates) in enumerate(ablation_configs.items()):
        print(f"\n{'='*60}")
        print(f"消融实验 {i+1}/{len(ablation_configs)}: {name}")
        print(f"{'='*60}\n")
        
        # 合并配置
        config = argparse.Namespace(**{**vars(base_config), **config_updates})
        
        # 为每个实验设置唯一的实验名称
        # 如果没有指定exp_name，则使用实验配置名称
        if not hasattr(config, 'exp_name') or not config.exp_name:
            # 清理名称，使其适合作为文件夹名
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')
            # 使用简洁的命名：直接使用实验名称
            if hasattr(base_config, 'exp_prefix') and base_config.exp_prefix:
                config.exp_name = f"{base_config.exp_prefix}_{safe_name}"
            else:
                config.exp_name = safe_name
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 训练模型
        try:
            model, vocab, exp_dir = train_model(config)
            
            # 检查是否成功
            if model is None or vocab is None or exp_dir is None:
                print(f"实验 '{name}' 跳过（数据文件不存在）")
                continue
            
            # 读取结果
            with open(os.path.join(exp_dir, 'results.json'), 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            result['experiment_name'] = name
            result['config'] = vars(config)
            results.append(result)
            
            print(f"\n实验 '{name}' 完成")
            print(f"最佳验证损失: {result['best_val_loss']:.4f}")
            
        except Exception as e:
            print(f"实验 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def plot_ablation_results(results, save_path):
    """绘制消融实验结果"""
    if not results:
        print("没有实验结果可绘制")
        return
    
    # 提取数据
    names = [r['experiment_name'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]
    best_val_losses = [r['best_val_loss'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x = range(len(names))
    width = 0.25
    
    # 损失对比
    axes[0].bar([i - width for i in x], train_losses, width, label='训练损失', alpha=0.8)
    axes[0].bar(x, val_losses, width, label='验证损失', alpha=0.8)
    axes[0].bar([i + width for i in x], best_val_losses, width, label='最佳验证损失', alpha=0.8)
    axes[0].set_xlabel('实验配置', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('消融实验结果对比', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 参数量对比
    param_counts = [r['num_params'] for r in results]
    axes[1].bar(names, param_counts, alpha=0.8, color='green')
    axes[1].set_xlabel('实验配置', fontsize=12)
    axes[1].set_ylabel('参数量', fontsize=12)
    axes[1].set_title('模型参数量对比', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, v in enumerate(param_counts):
        axes[1].text(i, v, f'{v/1e6:.2f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n消融实验结果图表已保存: {save_path}")


def create_results_table(results, save_path):
    """创建结果表格"""
    if not results:
        return
    
    data = []
    for r in results:
        data.append({
            '实验名称': r['experiment_name'],
            '最佳验证损失': f"{r['best_val_loss']:.4f}",
            '最终训练损失': f"{r['final_train_loss']:.4f}",
            '最终验证损失': f"{r['final_val_loss']:.4f}",
            '参数量': f"{r['num_params']/1e6:.2f}M"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"结果表格已保存: {save_path}")
    
    # 打印表格
    print("\n消融实验结果表格:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验')
    
    # 基础配置
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='data/tiny_shakespeare.txt')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_cuda', action='store_true', help='使用GPU')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--exp_prefix', type=str, default='', 
                       help='实验文件夹名称前缀（可选）')
    
    base_config = parser.parse_args()
    
    # 如果没有指定use_cuda，自动检测GPU
    if not base_config.use_cuda:
        base_config.use_cuda = torch.cuda.is_available()
    
    # 检查并下载数据文件
    data_path = base_config.data_path
    # 如果路径是相对路径，转换为相对于项目根目录的路径
    if not os.path.isabs(data_path):
        if 'data' not in data_path:
            data_dir = get_data_dir()
            filename = os.path.basename(data_path) or 'tiny_shakespeare.txt'
            data_path = os.path.join(data_dir, filename)
        else:
            data_dir = get_data_dir()
            filename = os.path.basename(data_path)
            data_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在，尝试下载...")
        data_dir = get_data_dir()
        filename = os.path.basename(data_path) or 'tiny_shakespeare.txt'
        downloaded_path = download_tiny_shakespeare(data_dir, filename)
        if downloaded_path:
            base_config.data_path = downloaded_path
        else:
            print("无法下载数据文件，请手动下载后重试")
            sys.exit(1)
    else:
        base_config.data_path = data_path
    
    # 定义消融实验配置
    ablation_configs = {
        'Baseline (完整模型)': {},
        '单层编码器': {'num_encoder_layers': 1, 'num_decoder_layers': 1},
        '无Dropout': {'dropout': 0.0},
        '小模型 (d_model=64)': {'d_model': 64, 'd_ff': 256, 'num_heads': 2},
        '大模型 (d_model=256)': {'d_model': 256, 'd_ff': 1024, 'num_heads': 8},
        '更多层 (4层)': {'num_encoder_layers': 4, 'num_decoder_layers': 4},
    }
    
    # 运行消融实验
    results = run_ablation_study(base_config, ablation_configs)
    
    # 保存结果
    results_base_dir = get_results_dir()
    results_dir = os.path.join(results_base_dir, 'ablation_study')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存JSON结果
    with open(os.path.join(results_dir, 'ablation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 绘制和保存图表
    plot_ablation_results(results, os.path.join(results_dir, 'ablation_comparison.png'))
    
    # 创建结果表格
    create_results_table(results, os.path.join(results_dir, 'ablation_results.csv'))
    
    print(f"\n消融实验完成！结果保存在: {results_dir}")

