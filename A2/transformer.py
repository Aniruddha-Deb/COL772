import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# rough sketch:
# Transformer
# - Embeddings
# - Position Embeddings
# -* EncoderLayer
#   -* MultiHeadAttention
#     -* SingleHeadAttention
#   -* Linear

class SingleHeadAttention(nn.Module):

    def __init__(self, embed_dim=512, att_dim=64, p=0.2):
        self.W_k = nn.Linear(embed_dim, att_dim)
        self.W_q = nn.Linear(embed_dim, att_dim)
        self.W_v = nn.Linear(embed_dim, att_dim)
        self.d   = np.sqrt(embed_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, X):
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)
        return self.dropout(F.softmax(Q@K.T/self.d, dim=-1)@V)

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads=8, embed_dim=512, att_dim=64):
        self.heads = [SingleHeadAttention(embed_dim=embed_dim, att_dim=att_dim) for i in range(n_heads)]
        self.linear = nn.Linear(att_dim*n_heads, embed_dim)

    def forward(self, X):
        att_concat = torch.hstack([att(X) for att in self.heads])
        return self.linear(att_concat)

class Sublayer(nn.Module):

    def __init__(self, module, size, p=0.2):
        self.module = module
        self.dropout = nn.Dropout(p)
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.norm(x+self.dropout(self.module(x)))

class FeedForwardNN(nn.module):

    def __init__(self, hidden_dim=2048, embed_dim=512, p=0.2):
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.d  = nn.Dropout(p)
        self.W2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        return self.W2(self.d(self.W1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, n_heads=8, embed_dim=512, att_dim=64):
        # TODO pass size
        self.attn = Sublayer(MultiHeadAttention(), size)
        self.ff = Sublayer(FeedForwardNN(), size)

    def forward(self, x):
        return self.ff(self.attn(x))

class Encoder(nn.Module):

    def __init__(self, n_layers=8):
        self.layers = [EncoderLayer() for i in range(n_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PositionalEmbedding(nn.Module):

    def __init__(self, dim=512, max_len=5000, p=0.2):

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x+self.pe[:,:x.shape[1]].requires_grad_(False))

