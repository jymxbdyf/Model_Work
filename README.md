# Transformer 手工实现与实验报告

本项目实现了完整的 Transformer 模型（包含 Encoder 和 Decoder），并在小规模文本建模任务上进行了训练和消融实验。

## 📁 项目结构

```
.
├── src/                    # 源代码目录
│   ├── model.py           # Transformer模型实现
│   ├── train.py           # 训练脚本
│   ├── ablation_study.py  # 消融实验脚本
│   ├── data_loader.py     # 数据加载工具
│   └── utils.py           # 工具函数
├── scripts/               # 运行脚本
│   └── run.sh            # 训练运行脚本
├── data/                  # 数据目录
│   └── tiny_shakespeare.txt  # 数据集
├── results/               # 实验结果
│   ├── exp_*/            # 每次实验的结果
│   └── ablation_study/   # 消融实验结果
├── requirements.txt      # Python依赖
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv transformer_env
source transformer_env/bin/activate  # Linux/Mac
# 或
transformer_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

下载 Tiny Shakespeare 数据集：

```bash
# 创建data目录
mkdir -p data

# 下载数据集（Linux/Mac）
wget -O data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Windows用户可以使用PowerShell
# Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -OutFile "data/tiny_shakespeare.txt"
```

或者手动下载：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

### 3. 训练模型

#### 基础训练

```bash
# 使用默认配置训练
python src/train.py --data_path data/tiny_shakespeare.txt --use_cuda

# 或使用运行脚本（Linux/Mac）
#bash scripts/run.sh
```

#### 完整训练命令（可重现实验）

```bash
python src/train.py \
    --d_model 128 \
    --num_heads 4 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --d_ff 512 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --epochs 20 \
    --seq_length 50 \
    --seed 42 \
    --grad_clip 1.0 \
    --scheduler cosine \
    --data_path data/tiny_shakespeare.txt \
    --use_cuda
```

#### 运行消融实验

```bash
python src/ablation_study.py \
    --data_path data/tiny_shakespeare.txt \
    --epochs 20 \
    --seed 42 \
    --use_cuda
```

## ⚙️ 参数说明

### 模型参数

- `--d_model`: 模型维度（默认：128）
- `--num_heads`: 注意力头数（默认：4）
- `--num_encoder_layers`: 编码器层数（默认：2）
- `--num_decoder_layers`: 解码器层数（默认：2）
- `--d_ff`: 前馈网络维度（默认：512）
- `--dropout`: Dropout率（默认：0.1）

### 训练参数

- `--batch_size`: 批次大小（默认：32）
- `--learning_rate`: 学习率（默认：3e-4）
- `--weight_decay`: 权重衰减（默认：0.01）
- `--epochs`: 训练轮数（默认：20）
- `--seq_length`: 序列长度（默认：50）
- `--seed`: 随机种子（默认：42）

### 训练稳定性技巧

- `--grad_clip`: 梯度裁剪阈值（默认：1.0）
- `--scheduler`: 学习率调度器（可选：cosine, step, warmup, none，默认：cosine）
- `--step_size`: StepLR步长（默认：10）
- `--gamma`: StepLR衰减率（默认：0.8）
- `--warmup_steps`: Warmup步数（默认：5）

## 🖥️ 硬件要求

- **CPU**: 支持即可运行（训练速度较慢）
- **GPU**: 推荐使用 CUDA 支持的 GPU（训练速度显著提升）
  - 显存要求：至少 2GB（batch_size=32, d_model=128）
  - 测试环境：NVIDIA GPU with CUDA 11.0+

## 📊 实验结果

训练完成后，结果会保存在 `results/exp_YYYYMMDD_HHMMSS/` 目录下：

- `best_model.pth`: 最佳模型检查点
- `final_model.pth`: 最终模型检查点
- `training_curves.png`: 训练曲线图
- `results.json`: 训练结果数据
- `config.json`: 实验配置
- `vocab.json`: 词汇表

### 消融实验结果

消融实验结果保存在 `results/ablation_study/` 目录下：

- `ablation_comparison.png`: 实验结果对比图
- `ablation_results.csv`: 结果表格
- `ablation_results.json`: 详细结果数据

## 🔬 实现特性

### 核心组件

✅ **Multi-Head Self-Attention**: 完整实现，包含缩放点积注意力机制  
✅ **Position-wise FFN**: 位置前馈网络，支持 GELU/ReLU 激活  
✅ **残差连接 + LayerNorm**: Pre-LN 架构  
✅ **位置编码**: 正弦位置编码  

### Encoder-Decoder 架构

✅ **Encoder Block**: 包含自注意力层和前馈网络层  
✅ **Decoder Block**: 包含掩码自注意力、交叉注意力和前馈网络层  
✅ **完整 Transformer**: 端到端的编码器-解码器模型  

### 训练稳定性技巧

✅ **AdamW 优化器**: 带权重衰减的 Adam 优化器  
✅ **学习率调度**: 支持 Cosine、Step、Warmup 调度器  
✅ **梯度裁剪**: 防止梯度爆炸  
✅ **Dropout**: 正则化防止过拟合  

### 实验功能

✅ **训练曲线可视化**: 自动生成损失和学习率曲线  
✅ **模型保存/加载**: 支持检查点保存和恢复  
✅ **参数统计**: 自动统计模型参数量  
✅ **消融实验**: 系统化的消融研究框架  

## 📝 代码说明

### 关键实现片段

#### Multi-Head Attention

```python
# 缩放点积注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

#### 位置编码

```python
# 正弦位置编码
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 残差连接 + LayerNorm

```python
# Pre-LN 架构
attn_output, _ = self.self_attn(x, x, x, mask)
x = self.norm1(x + self.dropout(attn_output))
```

详细实现请参考 `src/model.py`。

## 📚 数据集

本项目使用 **Tiny Shakespeare** 数据集，这是一个小规模的文本数据集，适合快速实验和验证。

- **来源**: https://github.com/karpathy/char-rnn
- **大小**: 约 1MB
- **内容**: 莎士比亚作品集

## 🔄 可重现性

所有实验都使用固定随机种子（默认：42）以确保可重现性。使用相同的命令和参数可以复现实验结果。


