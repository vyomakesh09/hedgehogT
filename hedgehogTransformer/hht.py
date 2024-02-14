import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Assuming HedgehogFeatureMap is defined as above
class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super(HedgehogFeatureMap, self).__init__()
        self.linear = nn.Linear(head_dim, head_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        # Apply the trainable linear transformation followed by an element-wise exp to x and its negation.
        return torch.cat(
            [torch.exp(self.linear(x)), torch.exp(-self.linear(x))], dim=-1
        )


class HedgehogAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(HedgehogAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.feature_map = HedgehogFeatureMap(self.head_dim)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.feature_map(self.values(values))
        keys = self.feature_map(self.keys(keys))
        queries = self.feature_map(self.queries(queries))

        # Einsum does matrix multiplication for query and keys for each training example
        # with optimization of space
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = HedgehogAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


'''class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        # Ensure pe is correctly shaped as [1, seq_length, embedding_dim]
        if self.pe.size(1) < seq_len:
            # Extend or regenerate pe to accommodate seq_len if needed
            # This is just a placeholder logic; actual implementation may vary based on your initial pe generation logic
            pass  # Implement logic to ensure pe covers seq_len
        
        pe = self.pe[:, :seq_len, :].to(x.device)
        
        print(f'x shape {x.shape}')
        print(f'pe shape: {pe[:, :seq_len, :].shape}')
        
        x = x + pe
        return x

        
       



class HedgehogTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(HedgehogTransformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, src_vocab_size)

    def forward(self, x):
        N, seq_length = x.shape
        positions = (
            torch.arange(0, seq_length).unsqueeze(0).repeat(N, 1).to(self.device)
        )

        out = self.dropout(self.word_embedding(x) + self.positional_encoding(positions))

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)
