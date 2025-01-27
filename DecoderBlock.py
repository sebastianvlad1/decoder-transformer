import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))

        # Feed-forward
        feed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward))

        return x
