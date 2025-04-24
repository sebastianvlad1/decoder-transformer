from datasets import load_dataset
from GPT import GPT
import torch
import torch.nn as nn
import torch.optim as optim
from GPTDataset import GPTDataset
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingLR

device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.backends.mps.is_available()
    else torch.device("mps")
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add a special token for padding if it doesnt exit
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def create_inputs_targets(examples):
    input_ids = torch.tensor(examples["input_ids"])
    inputs = input_ids[:, :-1]  # All tokens, except the last one
    targets = input_ids[:, 1:]  # All tokens, except the first one
    return {"inputs": inputs, "targets": targets}

# Apply the function on the entire dataset
processed_dataset = tokenized_dataset.map(create_inputs_targets, batched=True)

# Create a Dataset object for training
train_dataset = GPTDataset(processed_dataset)

num_batches = ceil(len(train_dataset) / 16)

# Create a DataLoader object
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
gpt2_model = GPT2Model.from_pretrained("gpt2")

model = GPT(gpt2_model, num_layers=12, heads=12, ff_hidden_size=3072, dropout=0.1, max_length=512)

model.to(device)
model.device = device


# Define loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)

epochs = 3

scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

model.train()

for epoch in range(epochs):  # Nr of epochs
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["inputs"].to(model.device)
        targets = batch["targets"].to(model.device)

        # Generate causal mask
        seq_length = inputs.size(1)
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool)).unsqueeze(0).to(device) # [1, seq_len, seq_len]
        padding_mask = (inputs != tokenizer.pad_token_id).unsqueeze(1).to(device) # [batch_size, 1, seq_len]
        combined_mask = causal_mask & padding_mask # broadcasting [batch_size, seq_len, seq_len]

        # Forward pass
        logits = model(inputs, mask=combined_mask)
        loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")
    scheduler.step()

torch.save(model.state_dict(), "gpt_model1.pth")