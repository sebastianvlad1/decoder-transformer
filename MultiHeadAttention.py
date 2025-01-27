import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Project values, keys, and queries to multi-head representation
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads chunks for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Rearrange dimensions to (heads, batch_size, seq_len, head_dim)
        values = values.permute(2, 0, 1, 3)
        keys = keys.permute(2, 0, 1, 3)
        queries = queries.permute(2, 0, 1, 3)

        # Compute energy scores using scaled dot-product attention
        energy = torch.einsum("hnqd,hnkd->hnqk", [queries, keys])  # (heads, batch, query_len, key_len)

        # Apply mask (if provided)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Compute attention weights
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # Weighted sum of values
        out = torch.einsum("hnqk,hnkd->hnqd", [attention, values]) # (heads, batch, query_len, head_dim)
        print(f"Shape of out before permute: {out.shape}")

        # Rearrange back to original shape: (batch_size, query_len, embed_size)
        out = out.permute(1, 2, 0, 3).contiguous()
        out = out.reshape(N, query_len, self.embed_size)

        # Apply final linear transformation
        out = self.fc_out(out)

        return out
