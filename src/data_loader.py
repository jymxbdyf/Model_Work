"""
数据加载和处理
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random


def build_vocab(text):
    """构建词汇表"""
    chars = sorted(list(set(text)))
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    vocab['<bos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    return vocab


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


def prepare_data(data_path, seq_length, train_ratio=0.9):
    """准备训练和验证数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 构建词汇表
    vocab = build_vocab(text)
    vocab_size = len(vocab)
    
    # 分割训练验证集
    chars = list(text)
    split_idx = int(train_ratio * len(chars))
    train_chars = chars[:split_idx]
    val_chars = chars[split_idx:]
    
    # 创建数据集（将文本分割成固定长度的序列）
    train_texts = []
    for i in range(0, len(train_chars), seq_length * 10):
        train_texts.append(train_chars[i:i + seq_length * 10])
    
    val_texts = []
    for i in range(0, len(val_chars), seq_length * 10):
        val_texts.append(val_chars[i:i + seq_length * 10])
    
    train_dataset = TextDataset(train_texts, vocab, seq_length)
    val_dataset = TextDataset(val_texts, vocab, seq_length)
    
    return train_dataset, val_dataset, vocab, vocab_size

