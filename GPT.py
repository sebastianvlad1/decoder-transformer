import torch.nn as nn
import torch
from Decoder import Decoder
from OutputLayer import OutputLayer

class GPT(nn.Module):
    def __init__(self, gpt2_model, num_layers, heads, ff_hidden_size, dropout, max_length):
        super(GPT, self).__init__()
        self.embed_size = gpt2_model.config.hidden_size
        self.vocab_size = gpt2_model.config.vocab_size

        # Token and position embeddings
        self.token_embedding = gpt2_model.wte
        self.position_embedding = gpt2_model.wpe

        # Decoder
        self.decoder = Decoder(self.embed_size, num_layers, heads, ff_hidden_size, dropout)

        # Output layer
        self.fc_out = OutputLayer(self.embed_size, self.vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape

        if mask is None:
            # Lower-triangular boolean mask (shape: [N, 1, seq_len, seq_len])
            mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            mask = mask.to(x.device)

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        # Embed tokens and positions
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + position_embeds)

        # Pass through decoder
        x = self.decoder(x, mask)

        # Map to vocabulary space
        logits = self.fc_out(x)

        return logits
