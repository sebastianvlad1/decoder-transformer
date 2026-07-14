def configure_tokenizer(tokenizer):
    """Use GPT-2 EOS as padding without changing the vocabulary size."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
