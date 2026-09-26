# LLM — training lab

A numbered, sequential lab for building language models from scratch. Each
folder is one stage; work through them in order. Notes on why a given
approach was chosen (and what was rejected) go in `decisions.md`.

## Folders

| Folder | Stage | What's in it now |
|---|---|---|
| `00-transformer-examples/` | Starting point. Self-contained PyTorch encoder + decoder examples with heavy comments. | `transformer_{encoder,decoder}_example.py` — see its own `README.md` |
| `01-build-gpt/` | Build a GPT from scratch — attention, blocks, training loop. | `bigram.py` baseline, `gpt.py` full model (MPS/CUDA/CPU, bf16 autocast, `torch.compile`, fused AdamW), `gpt-dev.ipynb` scratchpad, `input.txt` (tiny Shakespeare) |
| `02-gpt2-reproduce/` | Reproduce GPT-2 at small scale; measure against the published baseline. | `fineweb.py` (FineWeb-Edu → uint16 token shards), `train_gpt2.py` (GPT-2 124M, DDP-capable, cosine LR, val loss + sampling), `play.ipynb` |
| `03-tokenizer/` | Byte-pair encoding from scratch; vocabulary and merge behaviour. | `tokenization.ipynb`; running it trains a SentencePiece BPE model with a 400-token vocab (`tok400.model` / `.vocab`) from `toy.txt` -- all three are regenerable and gitignored |
| `04-post-training/` | SFT, preference tuning, and evaluation of the pretrained base. | notes only so far |
| `05-reading/` | Paper notes and derivations that back the folders above. | notes only so far |

| File | Role |
|---|---|
| `decisions.md` | Running log of design decisions and their rationale. |
| `<folder>/notes.md` | Per-stage working notes, with source attribution at the top. |

## Attribution

Folders are named for **what they contain**, not where the material came from —
`01-build-gpt`, never `01-<author>-build-gpt`. Attribution is content, so it
lives in the folder's `notes.md`, one line at the top:

> Code-along with <author>'s "<title>" (link). Typed by hand, not copied;
> deviations and experiments in `experiments/`.

That credits the source, states what's mine, and keeps paths accurate once my
own break-it experiments land in the folder.

## Environment

There is **one** environment for this repo, defined by `pyproject.toml` at the
repo root — no per-folder virtualenvs and no `requirements.txt`. The training
stack lives in the `llm` dependency group:

```bash
cd ..                      # repo root
uv sync --group llm        # or: uv sync --all-groups
```

**Run each script from its own folder.** Data paths are relative to the
working directory (`gpt.py` / `bigram.py` open `input.txt`,
`train_gpt2.py` reads `edu_fineweb10B/`), so `uv run python
LLM/01-build-gpt/gpt.py` from the repo root fails to find its data:

```bash
cd LLM/01-build-gpt && uv run python gpt.py

cd LLM/02-gpt2-reproduce
uv run python fineweb.py       # one-time: streams FineWeb-Edu, writes 3 x 100M-token shards (shard 0 = val)
uv run python train_gpt2.py    # single device; picks CUDA > MPS > CPU
# multi-GPU: uv run torchrun --standalone --nproc_per_node=<N> train_gpt2.py
```

`edu_fineweb10B/` is generated data and is gitignored.

Always run through the root environment (`uv run ...`) so every stage resolves
against the same `uv.lock`. Remote GPU work is the one exception: its
dependencies (CUDA torch, bitsandbytes, vllm) belong to a Modal container
image definition, not to this local environment. Local goes in
`pyproject.toml`; remote goes in the image.
