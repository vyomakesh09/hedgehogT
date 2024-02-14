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
        # Ensure the head dimension is halved if the feature map doubles it
        self.expanded_head_dim = (
            self.head_dim * 2
        )  # Assuming feature map doubles the dimension

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Adjust the output linear layer to account for the expanded head dimension
        self.fc_out = nn.Linear(heads * self.expanded_head_dim, embed_size)
        self.feature_map = HedgehogFeatureMap(self.head_dim)

    def forward(self, values, keys, query):
        # Forward method implementation remains the same until after feature map application
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Correct reshaping to [Batch, Seq_len, Heads, Head_dim]
        # Then, linear transformations are applied to each head separately
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        """print(
            f"values, keys, queries after reshaping: {values.shape, keys.shape, queries.shape}"
        )"""

        # After applying the feature map, the dimensionality of heads is now expanded
        values, keys, queries = [self.feature_map(x) for x in (values, keys, queries)]

        """print(
            f"values, keys, queries after applying feature map: {values.shape, keys.shape, queries.shape}"
        )"""

        # Attention calculation remains the same
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=-1)

        """print(
            f'attention after einsum of "nhql, nlhd->nqhd", [queries, keys] and softmax(attention / (self.embed_size ** (1 / 2)), dim=-1) attention, values in  hedgehog attention: {attention.size(), values.size()}'
        )"""

        # Apply attention to the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        """print(
            f'out after torch.einsum("nhql,nlhd->nqhd", [attention, values]): {out.shape}'
        )"""

        # Correctly reshape `out` to combine the expanded head dimensions back
        out = out.reshape(N, query_len, self.heads * self.expanded_head_dim)

        """print(
            f"out after reshape(N, query_len, self.heads * self.expanded_head_dim): {out.shape}"
        )"""

        # If necessary, apply a projection layer to reduce dimensionality back to `embed_size`
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


"""class PositionalEncoding(nn.Module):
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
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        """print(
            f"x.shape in positional encoding before pe broadcasted to matdch x batch size : {x.shape}"
        )"""

        seq_len = x.size(1)
        position = torch.arange(
            0, seq_len, dtype=torch.float, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Ensure pe is broadcasted to match x's batch size during addition
        pe = pe.expand(x.size(0), -1, -1)  # Adjust pe to match x's batch size

        """print(
            f"x.shape in positional encoding after pe broadcasted to matdch x batch size : {x.shape}"
        )
        print(f"pe shape after adjusted to match x batch size : {pe.shape}")"""

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
        x = self.dropout(self.word_embedding(x))  # Embedding
        x = x + self.positional_encoding(x)

        for layer in self.layers:
            out = layer(x, x, x)  # Process through Transformer blocks

        return self.fc_out(out)
