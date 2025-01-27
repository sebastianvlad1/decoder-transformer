from datasets import load_dataset
from GPT import GPT
import torch
import torch.nn as nn
import torch.optim as optim
from GPTDataset import GPTDataset
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader
from math import ceil

device = "mps" if torch.backends.mps.is_available() else "cpu"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#print(dataset)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add a special token for padding if it doenst exit
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#print(tokenized_dataset)

def create_inputs_targets(examples):
    input_ids = torch.tensor(examples["input_ids"])
    inputs = input_ids[:, :-1]  # Toate token-urile, mai puțin ultimul
    targets = input_ids[:, 1:]  # Toate token-urile, mai puțin primul
    return {"inputs": inputs, "targets": targets}

# Aplică funcția pe întregul dataset
processed_dataset = tokenized_dataset.map(create_inputs_targets, batched=True)
#print(processed_dataset)

# Creează un obiect Dataset pentru antrenare
train_dataset = GPTDataset(processed_dataset)

num_batches = ceil(len(train_dataset) / 16)
print(f"Number of rows in the train dataset: {len(train_dataset)}")
print(f"Number of batches in one epoch: {num_batches}")

# Creează un DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
gpt2_model = GPT2Model.from_pretrained("gpt2")

model = GPT(gpt2_model, num_layers=4, heads=4, ff_hidden_size=1024, dropout=0.1, max_length=512)

model.to(device)
model.device = device

# Loss și optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

model.train()
for epoch in range(1):  # Numărul de epoci
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["inputs"].to(model.device)
        targets = batch["targets"].to(model.device)

        # Forward pass
        logits = model(inputs, mask=None)
        loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

torch.save(model.state_dict(), "gpt_model.pth")