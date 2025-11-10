#!/bin/bash

# Transformer 训练运行脚本
# 使用方法: bash scripts/run.sh

echo "=========================================="
echo "Transformer 训练脚本"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "错误: 数据文件不存在"
    echo "请先下载数据集到 data/tiny_shakespeare.txt"
    echo "下载命令:"
    echo "  wget -O data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    exit 1
fi

# 创建必要的目录
mkdir -p results
mkdir -p data

# 检查是否使用GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，将使用 CUDA"
    USE_CUDA="--use_cuda"
else
    echo "未检测到 GPU，将使用 CPU"
    USE_CUDA=""
fi

echo ""
echo "开始训练..."
echo ""

# 运行训练（可重现实验的完整命令）
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
    $USE_CUDA

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果保存在 results/ 目录下"
echo "=========================================="

