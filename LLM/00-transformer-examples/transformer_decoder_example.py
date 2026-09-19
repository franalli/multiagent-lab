#!/usr/bin/env python3
"""
PyTorch Transformer Decoder Example (GPT-2 style) for Language Modeling
======================================================================

USE CASE:
---------
This example demonstrates how to build, train, and use a *decoder-only*
transformer -- the GPT-2 family of models -- for language modeling and text
generation, using PyTorch. It is the companion to
`transformer_encoder_example.py` and is meant to be read alongside it.

The encoder example took a whole sentence and produced ONE label
(classification). This decoder example does something different: it learns to
predict the **next token** given all previous tokens, and then *generates*
new text one token at a time. That single idea -- next-token prediction -- is
the entire training objective behind GPT-2, GPT-3, and modern chat LLMs.

ENCODER vs. DECODER -- THE ONE BIG DIFFERENCE:
----------------------------------------------
Architecturally a GPT-2 "decoder" is almost identical to the encoder example.
The defining difference is a single thing: a **causal (look-ahead) mask** that
stops every position from attending to tokens that come *after* it. With that
mask, position i can only use positions 0..i to predict token i+1, which is
exactly what you need to generate text left-to-right.

  Encoder  : every token sees every other token  (bidirectional)  -> understand
  Decoder  : every token sees only itself + past (causal)         -> generate

(Surprising-but-true detail addressed in the comments below: a decoder-only
model is built from `nn.TransformerEncoderLayer` + a causal mask, NOT from
`nn.TransformerDecoderLayer`. The latter adds cross-attention to an encoder's
output, which GPT-2 does not have.)

MODEL ARCHITECTURE:
-------------------
Summary:  Input ids -> Token Embedding (+ learned Positional Embedding)
          -> N x Transformer block (CAUSAL self-attention + MLP, pre-LayerNorm)
          -> Final LayerNorm -> LM head -> next-token logits at EVERY position

DETAILED LAYER-BY-LAYER SKETCH  (tensor shapes on the right; mirrors the code)

  Legend:  B=batch   S=seq_len (<= block_size)   D=d_model (128)   H=nhead (4)
           F=dim_feedforward (512)   V=vocab_size

  INPUT
    input_ids ................................. (B, S)     long
        |
        +--> Token Embedding       nn.Embedding(V, D) ............ (B, S, D)
        |
        +--> Positional Embedding  nn.Embedding(block_size, D) ... (S, D)  LEARNED
        |        (look up positions 0..S-1, then broadcast-add over the batch)
        v
    token + position  -->  Dropout ............ (B, S, D)
        |        (note: NO sqrt(D) scaling -- GPT-2 relies on its 0.02 init)
        |
        v   Build CAUSAL mask ................. (S, S)  0 on/below diag, -inf above
        |        (so position i may attend only to positions j <= i)
        v
  +-- Transformer Block  (PRE-LayerNorm) -- repeated x num_layers
  |
  |    in --> LayerNorm --> Masked Multi-Head Self-Attention (H heads, CAUSAL)
  |    +-------------------------------------------> ADD residual
  |
  |    in --> LayerNorm --> Feed-Forward: Linear(D->F) --> GELU --> Linear(F->D)
  |    +-------------------------------------------> ADD residual
  +--                                                        (B, S, D)
        |
        v   Final LayerNorm (ln_f) ............ (B, S, D)
        |
        v   LM Head   nn.Linear(D, V, bias=False)  [weight-TIED to Token Embedding]
    logits .................................... (B, S, V)
  OUTPUT   a next-token probability distribution at EVERY position
           (for generation: take logits[:, -1, :], sample, append, repeat)

MODEL INPUT/OUTPUT SHAPES:
-------------------------

Training (batched, teacher forcing):
  Input:
    - input_ids:  torch.Tensor shape (batch_size, seq_len)  dtype=torch.long
    - target_ids: torch.Tensor shape (batch_size, seq_len)  dtype=torch.long
                  (target is just input shifted LEFT by one position)
  Output:
    - logits:     torch.Tensor shape (batch_size, seq_len, vocab_size) float
                  A probability distribution over the next token AT EVERY
                  position -- not a single label like the encoder example.

Generation / inference (single sequence, autoregressive):
  Input:
    - idx:        torch.Tensor shape (1, prompt_len)               long
  Output:
    - idx:        torch.Tensor shape (1, prompt_len + new_tokens)  long
                  the prompt with newly generated tokens appended.

Where:
  - batch_size: number of sequences processed at once (e.g. 32)
  - seq_len:    length of each training block ("block_size"/context window)
  - vocab_size: number of distinct tokens (here: distinct characters)

EXAMPLE USAGE (illustrative -- exact ids depend on the learned vocabulary):
--------------------------------------------------------------------------
We use a *character-level* tokenizer, so one "token" is one character.

  A training window of block_size+1 = 11 chars taken from the corpus:

      window     :  "the cat sat"
      input_ids  :  "the cat sa"   ->  [t, h, e, _, c, a, t, _, s, a]
      target_ids :  "he cat sat"   ->  [h, e, _, c, a, t, _, s, a, t]
                                        ^ each input position predicts the
                                          NEXT character (target = input <<1)
                                        (_ shown for the space character)

  Generation from a prompt, one character at a time:

      prompt     :  "the "
      output     :  "the cat sat on the mat. the dog ran in the ..."
                    (illustrative; the actual text depends on training)

This is a complete, educational example demonstrating:
1. How a decoder-only (GPT-2) model differs from an encoder (just a mask!)
2. How to prepare data for next-token prediction (the "shift by one" trick)
3. How a causal mask works and why it makes generation possible
4. How to train with the language-modeling (cross-entropy) loss + perplexity
5. How to GENERATE text autoregressively with temperature sampling

Running this is NOT required to learn from it -- it is a reading artifact.
But it is written to run on plain local CPU (no GPU, no internet, no large
downloads) if you do want to try it.

The patterns shown here scale directly to real GPT-2 / GPT-3 / chat LLMs.
"""

import math
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Set random seed for reproducibility (so every run produces the same numbers)
SEED = 42
torch.manual_seed(SEED)

# Use GPU if available, otherwise CPU. This tiny model runs fine on CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================
#
# Language modeling needs *a lot* of text and a way to turn that text into
# integer ids. To keep this example dependency-free and self-contained we use:
#   - a tiny synthetic corpus (a handful of sentences, repeated), and
#   - a CHARACTER-LEVEL tokenizer (one character == one token).
#
# Character-level keeps the vocabulary tiny (~30 tokens) and lets us print
# real, human-readable generated text. Real GPT-2 uses a subword tokenizer
# (Byte-Pair Encoding) instead, but the training/generation logic is identical.
# =============================================================================


class CharTokenizer:
    """
    A minimal character-level tokenizer built from scratch (no libraries).

    A tokenizer is just a two-way mapping between text and integers:
      - encode: "hi" -> [list of int ids]   (so the model can do math on it)
      - decode: [list of int ids] -> "hi"   (so we can read the model's output)

    For character level, every distinct character in the corpus becomes one
    entry in the vocabulary. (The encoder example used whole *words*; the only
    difference here is the unit of tokenization.)
    """

    def __init__(self, text):
        """
        Build the vocabulary from a corpus string.

        Args:
            text: the full training corpus as one Python string.
        """
        # The vocabulary is the sorted set of unique characters in the corpus.
        # sorted() makes the mapping deterministic across runs.
        # chars: List[str], e.g. [' ', '.', 'a', 'b', 'c', ...]
        chars = sorted(set(text))

        # stoi = "string to int": maps a character -> its integer id
        # itos = "int to string": maps an integer id -> its character
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # vocab_size: number of distinct tokens the model must choose between.
        self.vocab_size = len(chars)

    def encode(self, s):
        """Convert a string to a list of integer ids. 'hi' -> [12, 17]"""
        # List[int] of length len(s)
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """Convert a list/iterable of integer ids back to a string."""
        # str of length len(ids)
        return "".join(self.itos[int(i)] for i in ids)


def create_synthetic_text(num_repeats=100):
    """
    Create a small synthetic corpus for demonstration.

    In a real project you would load megabytes/gigabytes of real text. Here we
    just repeat a handful of simple sentences so that:
      - the vocabulary stays tiny (only ~25 characters),
      - the patterns are easy for a small model to learn, and
      - the generated samples are recognizable as "English".

    Returns:
        A single string containing the whole corpus.
    """
    sentences = [
        "the cat sat on the mat. ",
        "the dog ran in the park. ",
        "a bird sang in the tree. ",
        "the sun set over the sea. ",
        "she read a good book. ",
        "we walked to the old town. ",
    ]
    # Concatenate the sentences once, then repeat the whole block num_repeats
    # times to get a longer stream of text to train on.
    return "".join(sentences) * num_repeats


class CharLanguageModelingDataset(Dataset):
    """
    Dataset for next-token (here: next-character) prediction.

    THE KEY IDEA -- "shift by one":
      A language model is trained so that, given tokens [0..i], it predicts
      token i+1. We get the training labels for free by simply taking the
      input sequence and shifting it left by one position.

      For a chunk of (block_size + 1) tokens:
        input_ids  = chunk[:-1]   (all but the last)
        target_ids = chunk[1:]    (all but the first)
      so target_ids[i] is the token that should come AFTER input_ids[i].

    Because every chunk has exactly block_size tokens, all samples are the same
    length -- so, unlike the encoder example, we need NO padding and NO padding
    mask here. The only mask the model uses is the causal mask (see PART 2).

    Each sample returned has:
      - input_ids:  shape (block_size,)  dtype=torch.long
      - target_ids: shape (block_size,)  dtype=torch.long
    """

    def __init__(self, data_ids, block_size):
        """
        Args:
            data_ids:   1D LongTensor of token ids for the whole corpus slice.
            block_size: context length -- how many tokens the model sees at once.
        """
        self.data = data_ids  # torch.Tensor shape (num_tokens,)
        self.block_size = block_size

    def __len__(self):
        # Every starting index from 0 .. len-block_size-1 yields one training
        # window of (block_size + 1) tokens, so this is how many windows exist.
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a contiguous chunk of (block_size + 1) tokens starting at idx.
        # chunk: torch.Tensor shape (block_size + 1,)
        chunk = self.data[idx : idx + self.block_size + 1]

        # input  = first block_size tokens; target = same window shifted by one.
        # x: (block_size,)   y: (block_size,)
        x = chunk[:-1]
        y = chunk[1:]

        return {
            "input_ids": x,  # Shape: (block_size,)
            "target_ids": y,  # Shape: (block_size,)  -- x shifted left by 1
        }


# =============================================================================
# PART 2: MODEL DEFINITION
# =============================================================================


def generate_causal_mask(seq_len, device):
    """
    Build the CAUSAL (look-ahead) mask -- the single thing that turns an
    encoder-style stack into a GPT-2 decoder.

    The mask is added to the attention scores BEFORE softmax. We put 0 where
    attention is allowed and -inf where it is forbidden. After softmax, the
    -inf positions become ~0 probability, so those tokens are effectively
    invisible to the current position.

    Rule: position i may attend to position j only if j <= i
          (i.e. a token can see itself and the past, never the future).

    For seq_len = 4 the returned matrix looks like (rows = query position i,
    columns = key position j):

            j=0    j=1    j=2    j=3
      i=0 [  0,   -inf,  -inf,  -inf ]   # token 0 sees only token 0
      i=1 [  0,    0,    -inf,  -inf ]   # token 1 sees tokens 0,1
      i=2 [  0,    0,     0,    -inf ]   # token 2 sees tokens 0,1,2
      i=3 [  0,    0,     0,     0   ]   # token 3 sees tokens 0,1,2,3

    Returns:
        mask: torch.Tensor shape (seq_len, seq_len), float, with 0 / -inf.

    (PyTorch ships an equivalent helper,
     `nn.Transformer.generate_square_subsequent_mask(seq_len)`; we build it by
     hand here so the mechanism is visible.)
    """
    # torch.triu keeps the upper triangle ABOVE the main diagonal (diagonal=1),
    # which is exactly the "future" positions we want to block with -inf.
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    return mask  # Shape: (seq_len, seq_len)


class GPTDecoderModel(nn.Module):
    """
    A small GPT-2 style, decoder-only transformer language model.

    Architecture (mirrors the encoder example, with the GPT-2 differences
    flagged in comments):
      Input ids -> Token Embedding + learned Positional Embedding
                -> N causal Transformer blocks (pre-LayerNorm, GELU MLP)
                -> Final LayerNorm
                -> LM head (Linear -> vocab) -> logits over the next token

    MODEL SHAPES:
    -------------
      input_ids:  (B, S)
        -> token_embedding            -> (B, S, d_model)
        -> + position_embedding       -> (B, S, d_model)
        -> transformer (causal mask)  -> (B, S, d_model)
        -> final layernorm            -> (B, S, d_model)
        -> lm_head                    -> (B, S, vocab_size)   <- logits

      Where B = batch_size, S = seq_len (<= block_size), d_model = embed dim.

    WHY nn.TransformerEncoderLayer FOR A "DECODER"?  (common point of confusion)
    ---------------------------------------------------------------------------
      A GPT-2 block is: causal self-attention -> MLP. That is precisely what
      `nn.TransformerEncoderLayer` computes -- we just feed it a causal mask.
      `nn.TransformerDecoderLayer` additionally performs CROSS-attention over a
      separate encoder's output ("memory"), which GPT-2 has no such thing as.
      So a *decoder-only* model correctly uses ENCODER layers + a causal mask.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size:      number of distinct tokens (output size of LM head).
            block_size:      maximum context length / number of positions.
            d_model:         embedding & model dimension.
            nhead:           number of self-attention heads (d_model % nhead == 0).
            num_layers:      number of stacked transformer blocks.
            dim_feedforward: hidden size of each block's MLP (GPT uses 4*d_model).
            dropout:         dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.block_size = block_size  # remembered so generation can crop context

        # 1. Token embedding: maps each token id -> a d_model vector.
        #    weight shape (vocab_size, d_model).  (B, S) ints -> (B, S, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional embedding (GPT-2 style: LEARNED, one vector per position).
        #    NOTE the contrast with the encoder example, which used a fixed
        #    sinusoidal formula. GPT-2 instead *learns* a position table of
        #    shape (block_size, d_model). This is why the model can only handle
        #    sequences up to block_size positions long.
        self.position_embedding = nn.Embedding(block_size, d_model)

        # Dropout applied to the summed embeddings (GPT-2 has this too).
        self.dropout = nn.Dropout(dropout)

        # 3. The transformer stack. One block = causal self-attention + MLP.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # GPT convention: 4 * d_model
            dropout=dropout,
            activation="gelu",  # GPT-2 uses GELU activations
            batch_first=True,  # shapes are (batch, seq, d_model)
            norm_first=True,  # PRE-LayerNorm: LN before attn/MLP.
            # GPT-2 uses pre-LN (more stable to
            # train) vs the original post-LN.
        )
        # Stack `num_layers` blocks. The `norm=` argument adds the final
        # LayerNorm that GPT-2 applies after the last block ("ln_f").
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 4. LM head: projects each position's d_model vector to a score for
        #    every token in the vocabulary. bias=False, like GPT-2.
        #    Input (B, S, d_model) -> Output (B, S, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # WEIGHT TYING (a real GPT-2 trick): share the same weight matrix
        # between the input token embedding and the output LM head. Both are
        # (vocab_size, d_model), so they can be the *same* parameter. This
        # saves parameters and usually improves quality. After this line they
        # are literally the same tensor -- initializing one initializes both.
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights the GPT-2 way: small Gaussian noise, std=0.02.

        Good initialization matters for stable training. GPT-2 draws embeddings
        and linear weights from a normal distribution with mean 0, std 0.02.
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # self.lm_head.weight is tied to token_embedding.weight (same tensor),
        # so it is already initialized -- no separate init needed.

    def forward(self, input_ids):
        """
        Run the model. Returns next-token logits for EVERY position.

        Args:
            input_ids: torch.Tensor shape (batch_size, seq_len), dtype long.

        Returns:
            logits: torch.Tensor shape (batch_size, seq_len, vocab_size), float.
        """
        # input_ids: (B, S) -- we only need the sequence length S here.
        S = input_ids.shape[1]

        # The learned position table only has `block_size` rows, so the model
        # physically cannot handle a longer sequence. This guard makes that
        # limit explicit (and is why generation crops to block_size below).
        assert S <= self.block_size, f"sequence length {S} exceeds block_size {self.block_size}"

        # Step 1: token embeddings.  (B, S) -> (B, S, d_model)
        tok_emb = self.token_embedding(input_ids)

        # Step 2: positional embeddings for positions [0, 1, ..., S-1].
        # positions: (S,) -> pos_emb: (S, d_model)
        positions = torch.arange(S, device=input_ids.device)
        pos_emb = self.position_embedding(positions)

        # Step 3: add them (broadcast over the batch) and apply dropout.
        # (B, S, d_model) + (S, d_model) -> (B, S, d_model)
        # NOTE: unlike the original Transformer / the encoder example, GPT-2
        # does NOT multiply the embeddings by sqrt(d_model); its 0.02 init
        # handles the scale instead.
        x = self.dropout(tok_emb + pos_emb)

        # Step 4: build the causal mask for this sequence length and run the
        # transformer stack. The mask is what makes attention left-to-right.
        # causal_mask: (S, S)
        causal_mask = generate_causal_mask(S, input_ids.device)
        # The `mask` argument is the additive attention mask (src_mask). We do
        # NOT pass a key-padding mask because there is no padding here.
        # x: (B, S, d_model) -> (B, S, d_model)
        x = self.transformer(x, mask=causal_mask)

        # Step 5: project every position to vocabulary logits.
        # (B, S, d_model) -> (B, S, vocab_size)
        logits = self.lm_head(x)

        return logits  # Shape: (B, S, vocab_size)


# =============================================================================
# PART 3: TRAINING SETUP
# =============================================================================


def collate_batch(batch):
    """
    Collate a list of samples into batched tensors for the DataLoader.

    Args:
        batch: list of dicts, each with:
               - input_ids:  (block_size,)
               - target_ids: (block_size,)

    Returns:
        dict with:
          - input_ids:  (batch_size, block_size)
          - target_ids: (batch_size, block_size)
    """
    # Stack the per-sample tensors along a new batch dimension (dim=0).
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    target_ids = torch.stack([item["target_ids"] for item in batch], dim=0)
    return {
        "input_ids": input_ids,  # Shape: (batch_size, block_size)
        "target_ids": target_ids,  # Shape: (batch_size, block_size)
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=5, patience=3):
    """
    Train the language model.

    The loop is the same shape as the encoder example, with two LM-specific
    twists:
      1. The loss compares logits at EVERY position against the shifted
         targets, so we flatten (B, S, V) -> (B*S, V) before cross-entropy.
      2. We report PERPLEXITY = exp(loss), the standard language-model metric.
         Lower is better; perplexity ~= "how many tokens the model is choosing
         between on average". (We also report next-token accuracy as an
         intuitive companion number.)

    Returns:
        The trained model (best validation weights) and a history dict.
    """
    model = model.to(DEVICE)

    best_val_loss = float("inf")  # for LM, LOWER loss is better
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": []}

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training mode enables dropout.
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_tokens = 0

        # ===== TRAINING LOOP =====
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)  # (B, S)
            target_ids = batch["target_ids"].to(DEVICE)  # (B, S)

            # Clear gradients left over from the previous step.
            optimizer.zero_grad()

            # Forward pass: predict next-token logits at every position.
            # logits: (B, S, V)
            logits = model(input_ids)
            B, S, V = logits.shape

            # Compute the language-modeling loss.
            # CrossEntropyLoss wants (N, num_classes) vs (N,), so we FLATTEN
            # the batch and sequence dims together: every one of the B*S
            # positions is an independent next-token classification problem.
            #   logits:  (B, S, V) -> (B*S, V)
            #   targets: (B, S)    -> (B*S,)
            loss = criterion(logits.reshape(B * S, V), target_ids.reshape(B * S))

            # Backward pass + gradient clipping (stabilizes transformer training).
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track running statistics (loss is a per-token mean, so weight it
            # by the number of tokens in this batch).
            num_tokens = target_ids.numel()  # B * S
            total_train_loss += loss.item() * num_tokens
            total_train_tokens += num_tokens

            # Next-token accuracy: did the argmax token match the target?
            # predicted: (B, S)
            predicted = logits.argmax(dim=-1)
            total_train_correct += (predicted == target_ids).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        epoch_train_loss = total_train_loss / total_train_tokens
        epoch_train_ppl = math.exp(epoch_train_loss)
        epoch_train_acc = total_train_correct / total_train_tokens

        # ===== VALIDATION LOOP =====
        model.eval()  # disables dropout
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_tokens = 0

        with torch.no_grad():  # no gradients needed for evaluation
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                target_ids = batch["target_ids"].to(DEVICE)

                logits = model(input_ids)  # (B, S, V)
                B, S, V = logits.shape
                loss = criterion(logits.reshape(B * S, V), target_ids.reshape(B * S))

                num_tokens = target_ids.numel()
                total_val_loss += loss.item() * num_tokens
                total_val_tokens += num_tokens
                predicted = logits.argmax(dim=-1)
                total_val_correct += (predicted == target_ids).sum().item()

        epoch_val_loss = total_val_loss / total_val_tokens
        epoch_val_ppl = math.exp(epoch_val_loss)
        epoch_val_acc = total_val_correct / total_val_tokens

        history["train_loss"].append(epoch_train_loss)
        history["train_ppl"].append(epoch_train_ppl)
        history["val_loss"].append(epoch_val_loss)
        history["val_ppl"].append(epoch_val_ppl)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train: Loss = {epoch_train_loss:.4f}, Perplexity = {epoch_train_ppl:.2f}, Acc = {epoch_train_acc:.4f}")
        print(f"  Val:   Loss = {epoch_val_loss:.4f}, Perplexity = {epoch_val_ppl:.2f}, Acc = {epoch_val_acc:.4f}")
        print(f"  Time:  {epoch_time:.2f}s")

        # Early stopping on validation loss (lower is better for LM).
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_gpt_model.pth")
            print(f"  New best model saved with val_loss = {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Restore the best-validation weights before returning.
    # weights_only=True is the safe choice: we are loading only tensors (a
    # state_dict), so there is no need to let torch unpickle arbitrary Python
    # objects -- which, from an untrusted .pth file, could execute code.
    model.load_state_dict(torch.load("best_gpt_model.pth", weights_only=True))
    return model, history


# =============================================================================
# PART 4: TEXT GENERATION (autoregressive inference)
# =============================================================================


@torch.no_grad()  # generation never needs gradients
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressively generate new tokens, one at a time.

    THE GENERATION LOOP (the heart of how GPT produces text):
      repeat max_new_tokens times:
        1. feed the current sequence through the model
        2. look ONLY at the logits for the last position (the next-token guess)
        3. turn those logits into probabilities (softmax)
        4. SAMPLE one token from that distribution
        5. append it to the sequence and loop

    Args:
        model:          a trained GPTDecoderModel.
        idx:            starting tokens, torch.Tensor shape (1, prompt_len), long.
        max_new_tokens: how many new tokens to produce.
        temperature:    >0 float. Scales logits before softmax. <1 makes the
                        model more confident/repetitive, >1 more random/creative,
                        1.0 = unchanged.
        top_k:          OPTIONAL. If set, sample only from the k most likely
                        tokens (a simple way to avoid rare, low-quality picks).
                        Leave as None for plain temperature sampling.

    Returns:
        idx: torch.Tensor shape (1, prompt_len + max_new_tokens), long.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # The model can only attend to block_size tokens, so if the running
        # sequence grew past that, crop to the most recent block_size tokens.
        # idx_cond: (1, <= block_size)
        idx_cond = idx[:, -model.block_size :]

        # Forward pass over the current context. logits: (1, T, vocab_size)
        logits = model(idx_cond)

        # We only care about the prediction for the FINAL position -- that is
        # the model's guess for the very next token.
        # logits[:, -1, :]: (1, vocab_size); divide by temperature to sharpen
        # or flatten the distribution.
        logits = logits[:, -1, :] / temperature

        # OPTIONAL top-k filtering: keep only the k highest-scoring tokens and
        # set the rest to -inf so they get ~0 probability after softmax.
        if top_k is not None:
            # v: (1, top_k) -- the top_k largest logits, sorted descending.
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # v[:, [-1]] is the smallest kept logit; anything below it is cut.
            logits[logits < v[:, [-1]]] = float("-inf")

        # Convert logits to a probability distribution over the vocabulary.
        # probs: (1, vocab_size)
        probs = F.softmax(logits, dim=-1)

        # SAMPLE one token id from the distribution (vs. always taking the
        # argmax, which would be deterministic and repetitive).
        # next_id: (1, 1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append the new token and continue.  (1, T) -> (1, T+1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=None,
    device=DEVICE,
):
    """
    Convenience wrapper: text prompt in, generated text out.

    Steps: encode the prompt -> run the generation loop -> decode back to text.

    NOTE: the prompt must use only characters that appeared in the training
    corpus (here: lowercase letters, space, and period). An unseen character
    has no id in the vocabulary and would raise a KeyError in encode().
    """
    model.eval()

    # Encode the prompt string into ids and add a batch dimension of 1.
    # idx: (1, prompt_len)
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Run the autoregressive loop.  out: (1, prompt_len + max_new_tokens)
    out = generate(model, idx, max_new_tokens, temperature=temperature, top_k=top_k)

    # Drop the batch dim and decode the full sequence (prompt + continuation).
    return tokenizer.decode(out[0].tolist())


def interactive_generation(model, tokenizer, max_new_tokens=200, temperature=0.8):
    """
    Interactive mode: type a prompt, watch the model continue it.

    This mirrors the encoder example's interactive inference, but instead of
    classifying your text it CONTINUES your text.
    """
    print("\n" + "=" * 60)
    print("LIVE TEXT GENERATION MODE")
    print("=" * 60)
    print("Type a prompt and the model will continue it (or 'quit' to exit).")

    while True:
        prompt = input("\nprompt> ")

        if prompt.lower() == "quit":
            break

        # Any character not seen during training has no id -- warn instead of
        # crashing (a small but real robustness point for char-level models).
        unknown = [c for c in prompt if c not in tokenizer.stoi]
        if unknown:
            print(f"  (skipping: these characters aren't in the vocabulary: {sorted(set(unknown))})")
            continue
        if not prompt:
            print("Please enter some text.")
            continue

        start_time = time.time()
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start_time

        print(f"\n{text}")
        print(f"\n(generated {max_new_tokens} tokens in {elapsed:.2f}s)")


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================


def main():
    """
    Tie everything together:
      1. Build the corpus, tokenizer, and datasets.
      2. Show a concrete input/target sample (the "shift by one" in action).
      3. Initialize the GPT decoder model.
      4. Train it with the language-modeling loss.
      5. Generate text from a few prompts, then go interactive.
    """
    print("PyTorch Transformer Decoder (GPT-2 style) Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Create data, tokenizer, and datasets
    # -------------------------------------------------------------------------
    print("\nStep 1: Creating and preparing data...")

    block_size = 64  # context length: how many characters the model sees at once
    batch_size = 32

    text = create_synthetic_text(num_repeats=100)
    tokenizer = CharTokenizer(text)

    print(f"  Corpus length: {len(text):,} characters")
    print(f"  Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"  Vocabulary: {''.join(tokenizer.itos[i] for i in range(tokenizer.vocab_size))!r}")

    # Encode the entire corpus into one long tensor of ids, then split 90/10
    # into train and validation regions (standard practice for LM data).
    data_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data_ids))
    train_ids, val_ids = data_ids[:n], data_ids[n:]

    train_dataset = CharLanguageModelingDataset(train_ids, block_size)
    val_dataset = CharLanguageModelingDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    print(f"  Training windows: {len(train_dataset):,}")
    print(f"  Validation windows: {len(val_dataset):,}")

    # ---- Show a concrete input/target sample: the "shift by one" in action ----
    sample = train_dataset[0]
    x_ids, y_ids = sample["input_ids"], sample["target_ids"]
    print("\n  Example training pair (note target = input shifted left by 1):")
    print(f"    input  ids [:12]: {x_ids[:12].tolist()}")
    print(f"    target ids [:12]: {y_ids[:12].tolist()}")
    print(f"    input  text: {tokenizer.decode(x_ids.tolist())!r}")
    print(f"    target text: {tokenizer.decode(y_ids.tolist())!r}")

    # -------------------------------------------------------------------------
    # STEP 2: Initialize the model
    # -------------------------------------------------------------------------
    print("\nStep 2: Initializing model...")

    model = GPTDecoderModel(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,  # = 4 * d_model, the GPT convention
        dropout=0.1,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    print(f"  Architecture: d_model=128, heads=4, layers=3, block_size={block_size}")

    # -------------------------------------------------------------------------
    # STEP 3: Set up training
    # -------------------------------------------------------------------------
    print("\nStep 3: Setting up training...")

    # CrossEntropyLoss is the language-modeling loss: at each position it scores
    # how well the predicted distribution matches the actual next token.
    criterion = nn.CrossEntropyLoss()

    # AdamW with a small learning rate is the standard GPT optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    print("  Optimizer: AdamW, lr=3e-4")
    print("  Loss: CrossEntropyLoss (next-token prediction)")

    # -------------------------------------------------------------------------
    # STEP 4: Train
    # -------------------------------------------------------------------------
    print("\nStep 4: Training model...")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=5,
        patience=3,
    )

    print("\nTraining complete!")
    print(f"Final validation perplexity: {history['val_ppl'][-1]:.2f}")

    # -------------------------------------------------------------------------
    # STEP 5: Generate text
    # -------------------------------------------------------------------------
    print("\nStep 5: Generating text from a few prompts...")

    for prompt in ["the ", "a bird ", "we walked "]:
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=120,
            temperature=0.8,
        )
        print(f"\n  prompt {prompt!r} ->")
        print(f"    {generated!r}")

    # Interactive mode (continues whatever you type).
    interactive_generation(model, tokenizer)

    print("\nDone!")


if __name__ == "__main__":
    main()
