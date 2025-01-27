import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return self.fc(x)
