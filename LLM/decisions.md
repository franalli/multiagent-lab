# Decisions

Running log: what was chosen, what was rejected, and why.

## 2026-09-17 — one root environment, `pyproject.toml` + uv groups

Replaced `requirements.txt` and `LLM/requirements.txt` with a single root
`pyproject.toml` and `uv.lock`. One file, one `.venv`, but dependency groups
(`dev`, `llm`, `audio`) let the agent stack install without the training stack
and vice versa.

Rejected: a second `requirements.txt` per subproject (the root file already
mixed `accelerate`, `datasets`, `modal` and `jupyter`, so splitting fought the
actual setup); per-folder virtualenvs (everything runs via `uv run` from root).

Kept out of scope: remote GPU dependencies (CUDA torch, bitsandbytes, vllm)
belong to a Modal container image definition, not to this local environment.

## 2026-09-17 — Python pinned to 3.12; numpy held at 2.3.5

`requires-python = ">=3.12,<3.13"`. Consequence worth knowing: `mistral-common`
caps `numpy<2.4` on Python <3.13, so the latest installable numpy here is
**2.3.5**, not 2.5.x. Widening to `>=3.13` is the single change that unlocks
numpy 2.5.x — nothing else in the tree objected.

## 2026-09-17 — all packages pinned to latest mutually-compatible versions

Versions were not hand-written; deps were left unpinned, `uv lock --upgrade`
resolved them, and the resolved set was pinned back. Notable bumps from the old
files: `librosa` 0.11.0 → **1.0.0** (major), `convex` 0.7.0 → 0.8.1,
`datasets` 4.8.5 → 5.0.1 (major), `modal` 1.4.3 → 1.5.5, `ruff` 0.15.13 → 0.16.8,
`pytest` → 9.1.1, `torch` 2.12 → 2.14.

`ruff` in the `dev` group and `rev:` in `.pre-commit-config.yaml` are kept in
lockstep at 0.16.8 — bumping one without the other means the hook and the venv
lint with different rule sets.

Side-effect of that bump: ruff 0.16 flags the teaching examples (shebang without
+x, spelled-out mutable defaults, nested `list()`). Rather than let a lint bump
rewrite code this commit only *moved*, `LLM/00-transformer-examples/` was added
to the existing ruff `exclude` — the same escape hatch `code-review/` already
uses for code that shouldn't be linted as production. The examples are therefore
byte-identical to their pre-move versions.

Dropped: `jupyter` metapackage → `jupyterlab` + `ipykernel`.

## 2026-09-17 — `torchtext` dropped; encoder example needs a small fix

`torchtext` came from `LLM/requirements.txt` and is **not** carried over. It is
archived upstream; 0.18.0 is the last release and its `libtorchtext.so` is built
against torch 2.3, so under torch 2.14 it resolves and installs but fails at
import:

    OSError: Could not load this library: .../torchtext/lib/libtorchtext.so

Verified with `uv run --with torchtext python -c "from torchtext.vocab import ..."`.
Pinning it into a group would install a package that cannot be imported, which
is worse than its absence.

**Open item (not done in this commit):**
`00-transformer-examples/transformer_encoder_example.py:113` imports
`build_vocab_from_iterator` from torchtext, so that example does not run as-is.
The fix is local to `build_vocabulary()` — a ~25-line `Vocab` stand-in
supporting `vocab[token]`, `vocab(tokens)`, `get_itos()` and a default index.
The decoder example is unaffected (it uses its own char tokenizer).
