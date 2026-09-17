# LLM — training lab

A numbered, sequential lab for building language models from scratch. Each
folder is one stage; work through them in order. Notes on why a given
approach was chosen (and what was rejected) go in `decisions.md`.

## Folders

| Folder | Stage |
|---|---|
| `00-transformer-examples/` | Starting point. Self-contained PyTorch encoder + decoder examples with heavy comments. |
| `01-build-gpt/` | Build a GPT from scratch — attention, blocks, training loop. |
| `02-gpt2-reproduce/` | Reproduce GPT-2 at small scale; measure against the published baseline. |
| `03-tokenizer/` | Byte-pair encoding from scratch; vocabulary and merge behaviour. |
| `04-post-training/` | SFT, preference tuning, and evaluation of the pretrained base. |
| `05-reading/` | Paper notes and derivations that back the folders above. |

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
uv run python LLM/01-build-gpt/gpt.py
```

Always run through the root environment (`uv run ...`) so every stage resolves
against the same `uv.lock`. Remote GPU work is the one exception: its
dependencies (CUDA torch, bitsandbytes, vllm) belong to a Modal container
image definition, not to this local environment. Local goes in
`pyproject.toml`; remote goes in the image.
