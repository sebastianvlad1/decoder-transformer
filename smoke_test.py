from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from GPT import GPT
from main import generate_text
from tokenizer_utils import configure_tokenizer


class DummyGPT2:
    def __init__(self, vocab_size, hidden_size, max_positions):
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_positions, hidden_size)


class TinyLM(nn.Module):
    def __init__(self, max_length=4, vocab_size=5):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.marker = nn.Parameter(torch.zeros(1))
        self.max_seen_length = 0

    def forward(self, input_ids, mask=None):
        self.max_seen_length = max(self.max_seen_length, input_ids.size(1))
        if input_ids.size(1) > self.max_length:
            raise AssertionError("generate_text exceeded model.max_length")
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size)
        logits[:, -1, 1] = 1.0
        return logits


class TinyTokenizer:
    eos_token_id = 0

    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.last_decoded_length = None

    def encode(self, prompt, return_tensors=None):
        ids = torch.arange(1, self.prompt_length + 1).unsqueeze(0)
        if return_tensors == "pt":
            return ids
        return ids.squeeze(0).tolist()

    def decode(self, ids, skip_special_tokens=True):
        self.last_decoded_length = len(ids)
        return " ".join(str(int(token_id)) for token_id in ids)


def build_model(vocab_size, max_length):
    gpt2_model = DummyGPT2(vocab_size=vocab_size, hidden_size=24, max_positions=max_length)
    model = GPT(
        gpt2_model,
        num_layers=1,
        heads=4,
        ff_hidden_size=48,
        dropout=0.0,
        max_length=max_length,
    )
    model.device = torch.device("cpu")
    return model


def test_tokenizer_uses_eos_as_pad_without_vocab_growth():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    original_vocab_size = len(tokenizer)

    configure_tokenizer(tokenizer)

    assert len(tokenizer) == original_vocab_size
    assert tokenizer.pad_token_id == tokenizer.eos_token_id


def test_padded_batch_runs_and_logits_shape_matches_vocab():
    tokenizer = configure_tokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
    model = build_model(vocab_size=len(tokenizer), max_length=8)
    encoded = tokenizer(
        ["hello", "hello world"],
        truncation=True,
        max_length=8,
        padding="max_length",
        return_tensors="pt",
    )

    logits = model(encoded["input_ids"], mask=None)

    assert logits.shape == (2, 8, model.vocab_size)


def test_forward_rejects_sequences_longer_than_max_length():
    model = build_model(vocab_size=32, max_length=8)
    too_long = torch.randint(0, model.vocab_size, (1, 9))

    try:
        model(too_long, mask=None)
    except ValueError as exc:
        assert "exceeds model max_length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized sequence")


def test_generate_text_clamps_new_tokens_to_context_window():
    model = TinyLM(max_length=4, vocab_size=5)
    tokenizer = TinyTokenizer(prompt_length=3)

    generate_text(model, tokenizer, "ignored", max_length=10, top_k=2)

    assert model.max_seen_length <= model.max_length
    assert tokenizer.last_decoded_length == model.max_length


def test_generate_text_rejects_prompt_longer_than_context_window():
    model = TinyLM(max_length=4, vocab_size=5)
    tokenizer = TinyTokenizer(prompt_length=5)

    try:
        generate_text(model, tokenizer, "ignored", max_length=1, top_k=2)
    except ValueError as exc:
        assert "exceeds model max_length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized prompt")


if __name__ == "__main__":
    test_tokenizer_uses_eos_as_pad_without_vocab_growth()
    test_padded_batch_runs_and_logits_shape_matches_vocab()
    test_forward_rejects_sequences_longer_than_max_length()
    test_generate_text_clamps_new_tokens_to_context_window()
    test_generate_text_rejects_prompt_longer_than_context_window()
    print("Smoke tests passed")
