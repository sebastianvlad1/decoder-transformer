from datasets import load_dataset
from GPT import GPT
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import GPT2Tokenizer, GPT2Model

# Încarcă dataset-ul WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Afișează structura dataset-ului
#print(dataset)

# Încarcă tokenizer-ul GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Adaugă un token special pentru padding (dacă nu există deja)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenizează textul
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Aplică tokenizarea pe întregul dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Afișează structura dataset-ului tokenizat
#print(tokenized_dataset)

def create_inputs_targets(examples):
    input_ids = torch.tensor(examples["input_ids"])  # Conversie explicită într-un tensor PyTorch
    inputs = input_ids[:, :-1]  # Toate token-urile, mai puțin ultimul
    targets = input_ids[:, 1:]  # Toate token-urile, mai puțin primul
    return {"inputs": inputs, "targets": targets}

# Aplică funcția pe întregul dataset
processed_dataset = tokenized_dataset.map(create_inputs_targets, batched=True)

# Afișează structura dataset-ului procesat
print(processed_dataset)


from torch.utils.data import Dataset, DataLoader

# Creează un Dataset customizat pentru PyTorch
class GPTDataset(Dataset):
    def __init__(self, processed_dataset):
        self.inputs = processed_dataset["train"]["inputs"]
        self.targets = processed_dataset["train"]["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx]),
            "targets": torch.tensor(self.targets[idx]),
        }

# Creează un obiect Dataset pentru antrenare
train_dataset = GPTDataset(processed_dataset)
from math import ceil
print(f"Number of rows in the train dataset: {len(train_dataset)}")
num_batches = ceil(len(train_dataset) / 4)

print(f"Number of batches in one epoch: {num_batches}")

# Creează un DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


gpt2_model = GPT2Model.from_pretrained("gpt2")
# Presupunem că ai definit modelul GPT într-o clasă numită `GPT`
model = GPT(gpt2_model, num_layers=6, heads=8, ff_hidden_size=2048, dropout=0.1, max_length=512)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.device = device

# Loss și optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Numărul de epoci
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


# def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
#     model.eval()
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
#     generated = input_ids
#
#     with torch.no_grad():
#         for _ in range(max_length):
#             logits = model(generated, mask=None)
#             next_token_logits = logits[:, -1, :] / temperature
#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
#             generated = torch.cat([generated, next_token], dim=-1)
#
#             # Oprire dacă se generează un token special (de ex., <end>)
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
#
#     return tokenizer.decode(generated[0], skip_special_tokens=True)
#
# # Exemplu de utilizare
# prompt = "Once upon a time"
# generated_text = generate_text(model, tokenizer, prompt, max_length=50)
# print(generated_text)