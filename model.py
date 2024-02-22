# model.py 
'''
reference
https://velog.io/@kwkim/Transformer%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84tensorflow  
https://www.youtube.com/watch?v=bCz4OMemCcA&t=1100s
'''

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
    mean = x.mean(-1, keepdim=True) # (batch, seq_len, 1)
    std = x.std(-1, keepdim=True) # (batch, seq_len, 1)
    return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

class TransformerEmbedding(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.positional_embedding = self._generate_positional_embedding(d_model, seq_len, dropout)

  def _generate_positional_embedding(self, d_model: int, seq_len: int, dropout: float):
    pe = torch.zeros(seq_len, d_model) # (Seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (Seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0) # (1, Seq_len, d_model)
    self.register_buffer('pe', pe)
    return pe

  def forward(self, x):
    token_embedding = self.token_embedding(x) * math.sqrt(self.d_model)
    positional_embedding = self.positional_embedding[:, :x.shape[1], :]
    return token_embedding + (positional_embedding).requires_grad_(False)


class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, n_head: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    assert d_model % n_head == 0, "d_model is not divisible by n_head"
    self.d_k = d_model // n_head 

    self.query = nn.Linear(d_model, d_model) 
    self.key = nn.Linear(d_model, d_model) 
    self.value = nn.Linear(d_model, d_model) 

    self.scale = math.sqrt(self.d_k)

    self.dense = nn.Linear(d_model, d_model) 
    self.dropout = nn.Dropout(dropout)

  def scaled_dot_product_attention(self, query, key, value, mask, dropout):
    # (Seq_len, d_k) -> (Seq_len, Seq_len)
    matmul_qk = query @ key.transpose(-2, -1)
    scaled_attention_logits = matmul_qk / self.scale

    if mask is not None: 
      scaled_attention_logits.masked_fill_(mask == 0, -1e9) # if mask == 0 fill it as -1e9 (sim -inf)
    
    attention_weights = scaled_attention_logits.softmax(dim=-1)
    attention_weights = dropout(attention_weights)
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


