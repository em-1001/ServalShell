# model.py

import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
  def __init__(self, features: int, eps: float=1e-6):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    # x: (batch, seq_len, hidden_size)
    mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
    std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
    return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerEmbedding(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.token_embedding = nn.Embedding(vocab_size, d_model).cuda()
    self.positional_embedding = self._generate_positional_embedding(d_model, seq_len, dropout).cuda()
    self.dropout = nn.Dropout(dropout)

  def _generate_positional_embedding(self, d_model: int, seq_len: int, dropout: float):
    pe = torch.zeros(seq_len, d_model).cuda() # (Seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).cuda() # (Seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float).cuda() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0) # (1, Seq_len, d_model)
    self.register_buffer('pe', pe)
    return pe

  def forward(self, x):
    token_embedding = self.token_embedding(x) * math.sqrt(self.d_model)
    positional_embedding = self.positional_embedding[:, :x.shape[1], :]
    embedded = token_embedding + (positional_embedding).requires_grad_(False)
    return self.dropout(embedded)


class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, n_head: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    assert d_model % n_head == 0, "d_model is not divisible by n_head"
    self.d_k = d_model // n_head

    self.query = nn.Linear(d_model, d_model, bias=False)
    self.key = nn.Linear(d_model, d_model, bias=False)
    self.value = nn.Linear(d_model, d_model, bias=False)

    self.scale = math.sqrt(self.d_k)

    self.dense = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def scaled_dot_product_attention(self, query, key, value, mask):
    # (Seq_len, d_k) -> (Seq_len, Seq_len)
    matmul_qk = query @ key.transpose(-2, -1)
    scaled_attention_logits = matmul_qk / self.scale

    if mask is not None:
      scaled_attention_logits.masked_fill_(mask == 0, -1e9) # if mask == 0 fill it as -1e9 (sim -inf)

    attention_weights = scaled_attention_logits.softmax(dim=-1)
    attention_weights = self.dropout(attention_weights)
    x = attention_weights @ value

    return x, attention_weights

  def split_heads(self, x, n_head, d_k):
    # (Batch, Seq_len, d_model) -> (Batch, Seq_len, n_head, d_k) -> (Batch, n_head, Seq_len, d_k)
    x = x.view(x.shape[0], x.shape[1], n_head, d_k).transpose(1, 2)
    return x

  def forward(self, query, key, value, mask):
    Q = self.query(query) # (Seq_len, d_model) -> (Seq_len, d_model)
    K = self.key(key) # (Seq_len, d_model) -> (Seq_len, d_model)
    V = self.value(value) # (Seq_len, d_model) -> (Seq_len, d_model)

    Q = self.split_heads(Q, self.n_head, self.d_k)
    K = self.split_heads(K, self.n_head, self.d_k)
    V = self.split_heads(V, self.n_head, self.d_k)

    x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

    # (Batch, n_head, Seq_len, d_k) -> (Batch, Seq_len, n_head, d_k) -> (Batch, Seq_len, d_model)
    concat_attention  = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)

    # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
    return self.dense(concat_attention)


class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float):
    super().__init__()
    self.ff_1 = nn.Linear(d_model, d_ff) # w1 and b1
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.ff_2 = nn.Linear(d_ff, d_model) # w2 and b2

  def forward(self, x):
    x = self.ff_2(self.dropout(self.relu(self.ff_1(x))))
    return x


class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (Batch, Seq_len, d_model) -> (Batch, Seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim = -1)


class EncoderLayer(nn.Module):
  def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
    super().__init__()

    self.layer_norm = LayerNormalization(d_model)
    self.self_attention = MultiHeadAttentionBlock(d_model, n_head, dropout)
    self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, src_mask):
    residual_1 = x
    x = self.self_attention(x, x, x, src_mask)
    x = self.dropout(x)
    x = self.layer_norm(residual_1 + x)

    residual_2 = x
    x = self.feed_forward(x)
    x = self.dropout(x)
    x = self.layer_norm(residual_2 + x)

    return x


class DecoderLayer(nn.Module):
  def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
    super().__init__()

    self.layer_norm = LayerNormalization(d_model)
    self.self_attention = MultiHeadAttentionBlock(d_model, n_head, dropout)
    self.cross_attention = MultiHeadAttentionBlock(d_model, n_head, dropout)
    self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    residual_1 = x
    x = self.self_attention(x, x, x, tgt_mask)
    x = self.dropout(x)
    x = self.layer_norm(residual_1 + x)

    residual_2 = x
    x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
    x = self.dropout(x)
    x = self.layer_norm(residual_2 + x)

    residual_3 = x
    x = self.feed_forward(x)
    x = self.dropout(x)
    x = self.layer_norm(residual_3 + x)

    return x


class Transformer(nn.Module):
  def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, n_head: int = 8, dropout = 0.1, d_ff = 2048):
    super().__init__()

    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.src_seq_len = src_seq_len
    self.tgt_seq_len = tgt_seq_len
    self.d_model = d_model
    self.N = N
    self.n_head = n_head
    self.d_ff = d_ff

    self.layer_norm = LayerNormalization(d_model)

    self.src_embed = TransformerEmbedding(src_vocab_size, d_model, src_seq_len, dropout)
    self.tgt_embed = TransformerEmbedding(tgt_vocab_size, d_model, tgt_seq_len, dropout)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(N)])
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(N)])

    self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

  def encode(self, src, src_mask):
    # (batch, seq_len, d_model)
    src = self.src_embed(src)

    for layer in self.encoder_layers:
      src = layer(src, src_mask)
    return self.layer_norm(src)

  def decode(self, tgt, encoder_output, src_mask, tgt_mask):
    # (batch, seq_len, d_model)
    tgt = self.tgt_embed(tgt)

    for layer in self.decoder_layers:
      tgt = layer(tgt, encoder_output, src_mask, tgt_mask)
    return self.layer_norm(tgt)

  def project(self, x):
    return self.projection_layer(x)
