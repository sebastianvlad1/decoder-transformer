# Decoder-Transformer: A Custom GPT Implementation

A simplified implementation of a GPT-style autoregressive language model using PyTorch. This project demonstrates core transformer architecture components and text generation capabilities, with a modular design for easy understanding and customization.

## Features

- Modular implementation of transformer decoder architecture
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Autoregressive text generation
- Training and inference scripts


## Installation

1. Clone the repository:
```bash
git clone git@github.com:sebastianvlad1/decoder-transformer.git
cd decoder-transformer
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training
To train the model, run the train.py script:
```bash
python train.py
```

## Text Generation
```bash
# main.py
# Example usage:
generated_text = generate_text(model, tokenizer, "Once upon a time", max_length=50)
print("Generated text:", generated_text)
```