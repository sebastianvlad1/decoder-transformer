import torch
from transformers import GPT2Tokenizer, GPT2Model
from GPT import GPT  # Import your custom GPT class

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the GPT model and its saved state
gpt2_model_config = {
    "num_layers": 12,
    "heads": 12,
    "ff_hidden_size": 3072,
    "dropout": 0.1,
    "max_length": 512,
}
device = "mps" if torch.backends.mps.is_available() else "cpu"
gpt2_model = GPT2Model.from_pretrained("gpt2")  # Load base GPT-2 model
model = GPT(gpt2_model, **gpt2_model_config)
model.load_state_dict(torch.load("gpt_model1.pth", map_location=device))  # Load weights
model.to(device)
model.device = device

# Define the text generation function
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)  # Tokenize and move to device
    generated = input_ids

    with torch.no_grad():  # Disable gradient calculations
        for _ in range(max_length):
            # Generate logits
            logits = model(generated, mask=None)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if <eos> token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(model, tokenizer, prompt, max_length=50)
print("Generated text:", generated_text)
