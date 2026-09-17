# PyTorch Transformer Examples

This folder contains a basic but complete example of training and running inference with a transformer-based encoder model in PyTorch. The example is designed for educational purposes to help understand how to use PyTorch for LLM training.

## Files

- `transformer_encoder_example.py` - Encoder example with model definition, training loop, and inference
- `transformer_decoder_example.py` - Decoder (GPT-style) counterpart: causal attention, char tokenizer

## Features

- Complete transformer encoder model implementation
- Tokenization and data loading pipeline
- Full training loop with optimization
- Live inference example
- Extensive comments explaining each step

## Quick Start

Dependencies come from the repo-root `pyproject.toml` (the `llm` group), not
from a local `requirements.txt`:

```bash
uv sync --group llm
uv run python LLM/00-transformer-examples/transformer_encoder_example.py
```

> **Status:** the decoder example runs as-is. The encoder example still imports
> `build_vocab_from_iterator` from `torchtext` (line 113), which is archived and
> cannot load against torch 2.14 — see `../decisions.md`. Replacing
> `build_vocabulary()` with a small local `Vocab` class is the outstanding fix.
