import torch.nn as nn
from DecoderBlock import DecoderBlock
class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
