"""
finetune.py

Supervised fine-tuning (SFT) of a Mistral base model with LoRA.

================================ SFT in one screen ================================
A *base* LLM only predicts "the next token" over generic web text. SFT teaches it
to follow instructions by showing it examples of (prompt -> ideal completion) and
nudging its weights so it makes those completions more likely. The training signal
is exactly the same next-token prediction it was pretrained with -- we just feed it
*our* curated examples and only score the part we care about (the completion).

The pipeline in this file:

  1. DATA      Load (prompt, completion) pairs and tokenize them into integer ids.
               Format each pair with the model's chat template, and mask the prompt
               tokens so the loss only rewards getting the *completion* right.   (SFTDataset)
  2. SPLIT     Hold out 10% as a validation set to watch for overfitting.        (get_loaders)
  3. ADAPT     Wrap the frozen 7B base with LoRA: tiny trainable matrices added
               to the attention layers. We train ~0.1% of the params, not 7B.   (LoraConfig)
  4. TRAIN     For each batch: forward -> loss -> backward -> optimizer step,
               resetting gradients every step.                                   (train)
  5. EVAL      After each epoch, measure loss on the held-out set with the model
               in eval mode and gradients off.                                   (evaluate)
  6. SAVE      Persist just the small LoRA adapter, not the whole base model.     (save_pretrained)

LoRA (Low-Rank Adaptation): instead of updating the giant weight matrices, we freeze
them and learn a small low-rank "delta" (rank r=8 here) bolted onto each attention
projection. Far less memory, and the result is a few-MB adapter you can swap on top
of the base model at inference time.
==================================================================================
"""

import json
import random
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)  # #16: log instead of bare print

MODEL = "mistralai/Mistral-7B-v0.1"
EPOCHS = 3
LR = 2e-4  # #13: 1e-3 is too high for LoRA on a 7B; 1e-4..2e-4 is the usual range
BATCH = 8
MAX_LEN = 512
SEED = 0  # #12: a single seed so the run is reproducible

# #14: pick the device once instead of hardcoding .cuda() (breaks on CPU/MPS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SFTDataset(Dataset):
    """Turns raw (prompt, completion) pairs into model-ready tensors.

    A PyTorch Dataset must answer two questions: "how many examples?" (__len__)
    and "give me example i" (__getitem__). The DataLoader then batches them.
    Here the real work is in __getitem__: text -> token ids + a label mask that
    tells the loss which tokens to actually learn from.
    """

    def __init__(self, path):
        # Load the corpus once and build the tokenizer (text <-> integer-id converter).
        # The tokenizer must match the model -- it defines the model's vocabulary.
        # #9: open in a `with` block so the file handle is closed
        with open(path) as f:
            self.rows = json.load(f)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        # A pad token is needed to make variable-length examples the same length so
        # they stack into a rectangular batch tensor.
        # #8: Mistral has no pad token by default, so padding="max_length" errors
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # Build one training example: input_ids (what the model reads), attention_mask
        # (which positions are real vs padding), and labels (what it should predict,
        # with -100 anywhere we don't want to score).
        row = self.rows[i]
        # #7: render through Mistral's chat template instead of raw concatenation,
        #     so the model sees the instruction format it was trained/served with.
        msgs = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        # Prompt-only render (with the generation prompt) marks where the completion starts.
        prompt_ids = self.tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
        # #8: truncation=True so long examples are cut instead of overflowing.
        full = self.tok.apply_chat_template(msgs, truncation=True, max_length=MAX_LEN)

        # WHAT ARE LABELS? labels[i] = the token the model should predict at position i.
        # The model reads tokens 0..i and predicts token i+1; HuggingFace shifts labels
        # by one internally, so `labels` starts as an exact COPY of input_ids ("the right
        # next token is just the next real token"). The special value -100 means "do not
        # score this position" in PyTorch's cross-entropy loss.
        #
        #   input_ids: [<user> what is 2+2? <asst>] [the answer is 4] [pad pad]
        #   labels:    [ -100  -100 ...        -100] [the answer is 4] [-100 -100]
        #              \------- prompt: ignored ----/\--- scored ---/ \-padding-/
        #
        # So the model still READS the prompt (context) but is only GRADED on generating
        # the completion -- that selective scoring is what makes it instruction-following.
        # #2: mask the prompt span to -100 so loss is computed on the completion only.
        prompt_len = min(len(prompt_ids), len(full))
        labels = list(full)  # start: labels == input_ids
        labels[:prompt_len] = [-100] * prompt_len  # blank out the prompt span

        # Pad to MAX_LEN so the default DataLoader collation works without a custom collator.
        pad_n = MAX_LEN - len(full)
        ids = full + [self.tok.pad_token_id] * pad_n
        mask = [1] * len(full) + [0] * pad_n  # #5: attention mask hides the padding
        labels = labels + [-100] * pad_n  # padding tokens ignored in the loss
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
        }


def get_loaders(path):
    """Build the train and validation DataLoaders.

    We hold out a slice of data the model never trains on. Train loss always
    drops; the *validation* loss is the honest signal -- if it stops improving
    while train loss keeps falling, the model is memorizing (overfitting).
    DataLoaders also handle batching and (for train) shuffling for us.
    """
    ds = SFTDataset(path)
    n = len(ds)
    n_val = max(1, int(n * 0.1))
    # #11: random (seeded) split instead of taking the first 90% by file order,
    #      which is non-reproducible and can be skewed/leaky.
    train, val = random_split(
        ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    # #10: shuffle the train loader (default is False) so order doesn't bias the run
    return (
        DataLoader(train, batch_size=BATCH, shuffle=True),
        DataLoader(val, batch_size=BATCH),
    )


def evaluate(model, loader):
    """Measure average loss on the held-out set -- our overfitting gauge.

    Two things differ from training: we put the model in eval mode (turns off
    dropout/regularization randomness so the number is stable and comparable),
    and we disable gradient tracking (we're only reading the model, not updating
    it, so there's no reason to spend memory/time building the autograd graph).
    """
    model.eval()  # #4: disable dropout for a stable val loss
    total = 0
    with torch.no_grad():  # #4: don't track grads during eval (saves memory/time)
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)  # #5: pass the mask
            labels = batch["labels"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total += out.loss.item()
    model.train()  # #4: restore train mode for the next epoch
    logging.info("val_loss=%.4f", total / len(loader))


def train(path):
    """The end-to-end fine-tuning run: load model, attach LoRA, loop, save.

    The core loop is the standard PyTorch four-step dance, repeated per batch:
      1. zero_grad  -- clear last step's gradients (PyTorch accumulates by default)
      2. forward    -- model(...) computes the loss for this batch
      3. backward   -- compute d(loss)/d(weights) for every trainable param
      4. step       -- the optimizer nudges those weights down the gradient
    Do this over the whole dataset once = one "epoch"; we run EPOCHS of them.
    """
    torch.manual_seed(SEED)  # #12: reproducibility
    random.seed(SEED)

    # Load the pretrained base model (the 7B of knowledge we're adapting, not replacing).
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    # #6: target the attention projections, not lm_head, and set the LoRA
    #     hyperparameters + task type so the adapter actually learns.
    config = LoraConfig(
        r=8,  # rank of the low-rank delta: each adapted weight W gets W + B@A
        #   where A is (r x in) and B is (out x r). Higher r = more
        #   trainable params / capacity; 8 is a common small default.
        lora_alpha=16,  # scaling factor: the delta is multiplied by alpha/r (here 2x).
        #   Lets you tune the adapter's strength independently of r.
        lora_dropout=0.05,  # dropout applied to the LoRA input during training; light
        #   regularization to reduce overfitting on small datasets.
        task_type="CAUSAL_LM",  # tells PEFT this is a decoder/next-token model so it wires the
        #   adapter in correctly (vs seq-classification, seq2seq, etc.).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # which sub-layers get an adapter: the attention query/key/value/
        #   output projections -- where the model decides what to attend
        #   to. (Not lm_head, the output vocab layer, which barely helps.)
    )
    model = get_peft_model(
        model, config
    )  # base weights are frozen here (correct as-is)
    model = model.to(DEVICE)  # #14: device-agnostic

    # Pass only the trainable (adapter) params to the optimizer; the frozen base
    # gets no grad anyway. AdamW is the standard choice for transformer fine-tuning.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    train_loader, val_loader = get_loaders(path)

    losses = []
    for epoch in range(EPOCHS):
        # One epoch = one full pass over the training data, batch by batch.
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)  # #5: pass the mask
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()  # #1: reset grads each step
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())  # #3: store the float, not the graph
        evaluate(model, val_loader)

    # #15: save just the LoRA adapter, not a multi-GB full state dict
    model.save_pretrained("adapter/")
