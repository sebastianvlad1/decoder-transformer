import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model

# Byte Pair Encoding
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "I love cats."

input_ids = tokenizer.encode(input_text, return_tensors="pt")

print("Input IDs:", input_ids)
print("Decoded Tokens:", tokenizer.decode(input_ids[0]))

# Load GPT2Model for embedding layer and positional encoding
gpt2_model = GPT2Model.from_pretrained("gpt2")

# Get the embeddings for the input IDs
with torch.no_grad():
    embeddings = gpt2_model.wte(input_ids)  # Word token embeddings
    position_embeddings = gpt2_model.wpe(torch.arange(input_ids.shape[1]).unsqueeze(0))  # Position embeddings
    input_embeddings = embeddings + position_embeddings  # Combine word and position embeddings

print("Input Embeddings Shape:", input_embeddings.shape)