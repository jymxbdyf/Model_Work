"""
Transformer模型实现
包含：Multi-Head Self-Attention, Position-wise FFN, 残差连接+LayerNorm, 位置编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """正弦位置编码实现
    
    数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制
    
    数学公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力
        
        Args:
            Q: [batch_size, num_heads, q_len, d_k]
            K: [batch_size, num_heads, k_len, d_k]
            V: [batch_size, num_heads, v_len, d_k]
            mask: [batch_size, 1, q_len, k_len] 或 None
        Returns:
            output: [batch_size, num_heads, q_len, d_k]
            attn_weights: [batch_size, num_heads, q_len, k_len]
        """
        # 计算注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            # mask中，-inf表示需要mask的位置，0表示不需要mask的位置
            # 直接将mask加到scores上（-inf + scores = -inf）
            scores = scores + mask
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, q_len, d_model] 查询序列
            K: [batch_size, k_len, d_model] 键序列（可能与Q长度不同）
            V: [batch_size, v_len, d_model] 值序列（通常与K长度相同）
            mask: [q_len, k_len] 或 [batch_size, q_len, k_len] 或 None
        Returns:
            output: [batch_size, q_len, d_model]
            attn_weights: [batch_size, num_heads, q_len, k_len]
        """
        batch_size, q_len = Q.size(0), Q.size(1)
        k_len = K.size(1)
        v_len = V.size(1)
        
        # 线性变换并分头
        # Q: [batch_size, q_len, d_model] -> [batch_size, num_heads, q_len, d_k]
        Q = self.w_q(Q).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, k_len, d_model] -> [batch_size, num_heads, k_len, d_k]
        K = self.w_k(K).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, v_len, d_model] -> [batch_size, num_heads, v_len, d_k]
        V = self.w_v(V).view(batch_size, v_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 扩展掩码维度
        if mask is not None:
            if mask.dim() == 2:  # [q_len, k_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            elif mask.dim() == 3:  # [batch_size, q_len, k_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, q_len, k_len]
            # 如果已经是4维，保持不变
        
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头: [batch_size, num_heads, q_len, d_k] -> [batch_size, q_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model)
        
        # 输出投影
        output = self.w_o(attn_output)
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    """位置前馈网络
    
    数学公式：
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    或使用GELU: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层
    
    包含：
    1. Multi-Head Self-Attention + 残差连接 + LayerNorm
    2. Position-wise FFN + 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差 + 层归一化 (Pre-LN架构)
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层
    
    包含：
    1. Masked Multi-Head Self-Attention + 残差连接 + LayerNorm
    2. Multi-Head Cross-Attention + 残差连接 + LayerNorm
    3. Position-wise FFN + 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_len, d_model] 解码器输入
            enc_output: [batch_size, src_len, d_model] 编码器输出
            src_mask: [batch_size, src_len, src_len] 或 None
            tgt_mask: [batch_size, tgt_len, tgt_len] 或 None (因果掩码)
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # 自注意力 + 残差 + 层归一化
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力 + 残差 + 层归一化
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型（Encoder-Decoder架构）"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4,
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=512, 
                 max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（下三角矩阵）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch_size, src_len] 源序列
            tgt: [batch_size, tgt_len] 目标序列
            src_mask: [batch_size, src_len, src_len] 或 None
            tgt_mask: [batch_size, tgt_len, tgt_len] 或 None
        Returns:
            [batch_size, tgt_len, tgt_vocab_size]
        """
        # 编码器
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        enc_output = src_embedded
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        dec_output = tgt_embedded
        
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.output_layer(dec_output)
        return output
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

