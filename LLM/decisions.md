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

## 2026-09-17 — ruff 0.16 broadened the default rule set; lint policy now declared

The `ruff` 0.15.13 → 0.16.8 bump surfaced **57 findings in untouched code**. The
cause is not the code: ruff 0.16 ships a much broader default rule set. Verified
with `ruff check --isolated` on a 7-line file, which flags `BLE001`, `S110`,
`B006` and `DTZ005` with no config present at all.

The repo had no `[tool.ruff]` section, so its lint policy was silently whatever
the installed ruff defaulted to. It is now declared in `pyproject.toml`:

* `exclude = ["code-review", "modal-examples-main"]` — mirrors the pre-commit
  excludes so a bare `ruff check .` agrees with the hook. `code-review/` holds
  intentionally-flawed fixtures; `modal-examples-main/` is vendored and carries
  its own config.
* `[tool.ruff.lint.isort] known-first-party` lists the `modal/` sibling modules.
  Without it isort merged `from common import ...` into the third-party block,
  contradicting the documented "modal/ is intentionally not a package"
  convention. Declaring them first-party restores the separation.

**Still recommended:** pin `lint.select` explicitly. Until then the next ruff
bump can change what fails again, the same way this one did.

All 57 were cleared rather than suppressed, except where the behaviour is
deliberate and now annotated house-style (`# noqa: <rule> -- <reason>`):
blind excepts in agent loops (tool errors go back to the model, never kill the
loop), and the documented fail-open in `modal/common.py`. Real fixes included
timezone-aware `datetime.now(UTC)` in the circuit breaker (re-ran it: the
CLOSED → OPEN → HALF_OPEN → CLOSED path still works), a mutable default
argument, and two nested `async with` merges.

One autofix regression worth knowing: `RUF100` deletes "unused" `# noqa`
comments **including their trailing prose**. Three explanations documenting real
constraints were lost and restored as plain comments — notably
`import agent_tools  # deferred: module path only valid after writing it`, which
encodes convention #5 (agent_tools is materialised at `/tmp` before import).

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
