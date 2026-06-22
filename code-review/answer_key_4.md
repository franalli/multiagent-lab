# Answer Key — `review_me_4.py` (variant 4, fine-tuning)

**Open only after your 30-minute review.** Same protocol as `how_to_run.md`: 30 minutes, no AI, narrate aloud, review and improve.

This is a PyTorch LoRA/SFT fine-tuning script. The review is **engineering correctness, not theory**: the bugs are the kind that silently ruin a run or break it outright. They map one-to-one to the "ML / training-script review" block in the guide. If you get a training script in the round, this is the rep.

---

## Tier 1 — Training-loop correctness (silently ruins or breaks the run)

1. **[MUST-CATCH] Missing `optimizer.zero_grad()`.** The loop does `backward()` then `step()` but never zeroes grads, so gradients accumulate across every step of every epoch. Each update is corrupted by all prior steps. This is the single most important bug here. Fix: `optimizer.zero_grad()` before each `backward()`.
2. **[MUST-CATCH] Loss not masked over the prompt.** `__getitem__` returns `labels = input_ids`, so the model is trained to predict the prompt tokens, not just the completion. For SFT you mask the prompt span to `-100` so it is ignored by the loss. As written, this is barely instruction tuning. Fix: build labels with the prompt positions set to `-100`.
3. **[MUST-CATCH] Appending the loss tensor leaks memory.** `losses.append(loss)` keeps each step's full autograd graph alive, so memory grows every step until OOM. Fix: `losses.append(loss.item())` (a Python float, no graph), or `.detach()`.
4. **[MUST-CATCH] `evaluate` has no `model.eval()` and no `torch.no_grad()`.** Dropout stays active, so the validation loss is noisy, and gradients are tracked through eval, wasting memory and time (and you never `model.train()` back). Fix: `model.eval()` and wrap the loop in `with torch.no_grad():`, then restore `model.train()`.
5. **[MUST-CATCH] No `attention_mask` passed.** Sequences are padded to `max_length`, but the model is called with only `input_ids`, so it attends to pad tokens. Fix: keep the tokenizer's `attention_mask` and pass it to the model.

## Tier 1 — Fine-tuning specifics (LoRA / Mistral)

6. **[MUST-CATCH] Wrong LoRA target modules.** `target_modules=["lm_head"]` adapts the output head, not the transformer. For a Mistral decoder you target the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, sometimes the MLP). As written, the adapter learns almost nothing useful. Also set `lora_alpha`, `lora_dropout`, and `task_type="CAUSAL_LM"`.
7. **No chat template.** `row["prompt"] + row["completion"]` concatenates raw text and ignores Mistral's instruction format. Use `tokenizer.apply_chat_template(...)` so the model sees the format it was trained and served with.
8. **Pad token unset; padding/truncation.** Mistral's tokenizer has no `pad_token` by default, so `padding="max_length"` will error; set `tok.pad_token = tok.eos_token`. And there is no `truncation=True`, so long examples are not cut.

## Tier 2 — Data pipeline

9. **Whole dataset into memory; file not closed.** `json.load(open(path))` reads everything and leaks the handle. For non-trivial data, stream (JSONL line by line) and use a `with` block.
10. **`shuffle=False` on the train loader.** `DataLoader(train, batch_size=BATCH)` defaults to no shuffling, which hurts convergence and lets data order bias the run. Fix: `shuffle=True` for train (leave val unshuffled).
11. **Ordered split, no shuffle or seed.** The 90/10 split takes the first 90% of the file as train and the last 10% as val by order, so the split is non-random, potentially skewed or leaky, and not reproducible. Fix: a seeded random split.
12. **No seed anywhere.** Nothing sets `torch.manual_seed` / `random.seed`, so runs are not reproducible.

## Tier 2 — Config, device, checkpointing

13. **Learning rate too high.** `1e-3` is high for LoRA fine-tuning of a 7B; `1e-4` to `2e-4` is the usual range. (Plus the hyperparameters are all hardcoded magic numbers.)
14. **Hardcoded `.cuda()` with no device check.** Breaks on CPU or MPS. Fix: `device = "cuda" if torch.cuda.is_available() else "cpu"` and `.to(device)` everywhere.
15. **Checkpoint saves only the full state dict.** `torch.save(model.state_dict(), ...)` saves no optimizer/epoch/scheduler state (cannot resume), and for LoRA you should save the adapter with `model.save_pretrained(...)`, not a multi-GB full state dict. There is also no eval-driven checkpointing or early stopping.
16. **`print` instead of logging; crude metric.** Minor: validation loss is printed, not logged, and there is no best-checkpoint selection.

## What NOT to flag (calibration)

- **The base is frozen.** `get_peft_model(model, config)` freezes the base weights and trains only the adapter, so "the base is not frozen" is wrong here. Passing `model.parameters()` (frozen plus trainable) to the optimizer is mildly wasteful, not a bug, since frozen params get no grad; a clean version filters to `p for p in model.parameters() if p.requires_grad`.

---

## Scoring bands

- **Strong.** Caught the missing `zero_grad` (#1), the unmasked prompt loss (#2), the loss-tensor memory leak (#3), the eval-mode/`no_grad` gap (#4), the missing attention mask (#5), and the wrong LoRA target modules (#6). Recognized the base is already frozen (did not false-flag it). Led with the training-loop bugs and the masking.
- **Pass.** Caught `zero_grad`, the loss-tensor leak, the eval-mode gap, and the wrong target modules, and noticed the labels are not masked, even if the masking refactor was rough.
- **Below bar.** Reviewed it as generic Python and missed that grads never reset, the model trains on the prompt, and eval runs with grads on.

## What a strong review sounds like (model opening)

> "The training loop has three bugs that ruin the run. There is no `optimizer.zero_grad()`, so gradients accumulate across steps; the labels are the full input, so it trains on the prompt instead of just the completion, which needs `-100` masking; and `losses.append(loss)` keeps the graph alive and will OOM, it should be `.item()`. Then `evaluate` has no `model.eval()` or `torch.no_grad()`, and the model is called with no `attention_mask` despite padding. On the LoRA side, the target modules are `lm_head` instead of the attention projections, so the adapter barely learns. The base is correctly frozen by `get_peft_model`, so that part is fine. Let me show the masking and the train-loop fix."

---

## Refactor models (the "improve" half)

**Tokenize with the chat template, truncation, pad token, and prompt masking:**

```python
class SFTDataset(Dataset):
    def __init__(self, path: str):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f]   # stream, handle closed
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __getitem__(self, i):
        row = self.rows[i]
        msgs = [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}]
        prompt_ids = self.tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
        full = self.tok.apply_chat_template(
            msgs, truncation=True, max_length=MAX_LEN)
        labels = list(full)
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)   # mask the prompt
        return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
        # (pad/collate to the batch max with a collator; pad labels with -100)
```

**Correct LoRA config:**

```python
config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Device, trainable-only optimizer, sane LR, seed:**

```python
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=2e-4)
```

**Train loop and eval done right:**

```python
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())          # scalar, no graph
    validate(model, val_loader, device)

def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            total += model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
    model.train()
    return total / len(loader)
```

**Save the adapter, not a full state dict:**

```python
model.save_pretrained("adapter/")               # LoRA weights only
```

The honest one-liner to land: the base is frozen and the loop runs, but as written it never resets gradients, trains on the prompt, leaks memory, and adapts the wrong modules, so it would produce a broken model that looks like it trained.
