#!/usr/bin/env python3
"""
PyTorch Transformer Encoder Example for Language Tasks
======================================================

USE CASE:
---------
This example demonstrates how to build, train, and use a transformer-based encoder
model for text classification using PyTorch. It's designed for learning the fundamentals
of LLM training pipelines.

The model takes text as input and outputs a classification prediction (e.g., sentiment
analysis: POSITIVE/NEGATIVE). While this example uses a simple binary classification
task, the architecture and training patterns apply directly to more complex language
modeling tasks.

MODEL ARCHITECTURE:
-------------------
Summary:  Input -> Token Embedding -> Positional Encoding -> Transformer Encoder
          -> Pooling -> Classifier

DETAILED LAYER-BY-LAYER SKETCH  (tensor shapes on the right; mirrors the code)

  Legend:  B=batch   S=seq_len   D=d_model (128)   H=nhead (4)
           F=dim_feedforward (512)   V=vocab_size   C=num_classes (2)

  INPUT
    input_ids ................................. (B, S)     long
    attention_mask ............................ (B, S)     long   1=token 0=pad
        |
        v   Token Embedding        nn.Embedding(V, D)
    token vectors ............................. (B, S, D)
        |
        v   Scale                  x * sqrt(D)
        v   Positional Encoding    sinusoidal, FIXED (not learned), + dropout
    position-aware vectors .................... (B, S, D)
        |
        v
  +-- Transformer Encoder Layer  (POST-LayerNorm) -- repeated x num_layers
  |
  |    in --> Multi-Head Self-Attention  (H heads, BIDIRECTIONAL)
  |    |         key-padding mask hides padded positions
  |    +---------------------> ADD residual --> LayerNorm
  |
  |    in --> Feed-Forward:  Linear(D->F) --> GELU --> Linear(F->D)
  |    +---------------------> ADD residual --> LayerNorm
  +--                                                        (B, S, D)
        |
        v   output of final layer ............. (B, S, D)
        |
        v   Mean Pooling   permute --> AdaptiveAvgPool1d(1) --> squeeze
    pooled "sentence" vector .................. (B, D)
        |
        v   Classifier             nn.Linear(D, C)
    logits .................................... (B, C)
  OUTPUT   one score per class  ->  argmax = predicted label

MODEL INPUT/OUTPUT SHAPES:
-------------------------

Training (batched):
  Input:
    - input_ids:      torch.Tensor shape (batch_size, seq_len)     dtype=torch.long
      Example: tensor([[   2,   5,  10, ...,   1],   # First text in batch
                       [   3,   7,   9, ...,   1],   # Second text in batch
                       ...])
    - attention_mask: torch.Tensor shape (batch_size, seq_len)     dtype=torch.long
      Example: tensor([[1, 1, 1, ..., 1],
                       [1, 1, 1, ..., 0],
                       ...])  # 1=real token, 0=padding
  Output:
    - logits:         torch.Tensor shape (batch_size, num_classes) dtype=torch.float
      Example: tensor([[2.5, -1.2],   # First sample: class 0 score=2.5, class 1 score=-1.2
                       [1.1, 3.4],    # Second sample
                       ...])

Inference (single sample):
  Input:
    - input_ids:      torch.Tensor shape (1, seq_len)          dtype=torch.long
    - attention_mask: torch.Tensor shape (1, seq_len)          dtype=torch.long
  Output:
    - logits:         torch.Tensor shape (1, num_classes)      dtype=torch.float

Where:
  - batch_size: Number of texts processed simultaneously (e.g., 16, 32, 64)
  - seq_len:    Maximum sequence length (e.g., 128, 256, 512)
  - num_classes: Number of classification categories (e.g., 2 for binary)

EXAMPLE USAGE:
--------------
Text: "happy joy love movie" -> Tokenized -> Embedded -> Transformer -> Prediction
Input shape:  (1, 8)  (1 text, 8 tokens including special tokens)
Output shape: (1, 2)  (1 text, 2 classes: NEGATIVE=0, POSITIVE=1)

This is a complete, educational example demonstrating:
1. How to define a transformer-based encoder model in PyTorch
2. How to prepare text data with tokenization and vocabulary building
3. How to train the model with proper batching and masking
4. How to run live inference with a trained model

The patterns shown here scale to larger models and more complex tasks.
"""

import math
import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.text.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check if CUDA (GPU) is available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================


class TextClassificationDataset(Dataset):
    """
    Custom Dataset class for text classification.

    In PyTorch, Dataset classes provide a way to:
    - Store your data
    - Access individual samples by index
    - Apply transformations to samples

    The __len__ method tells PyTorch how many samples are in the dataset.
    The __getitem__ method returns a single sample given its index.

    Each sample returned has:
      - input_ids:     shape (seq_len,)      dtype=torch.long
      - attention_mask: shape (seq_len,)      dtype=torch.long
      - label:          shape ()              dtype=torch.long
    """

    def __init__(self, texts, labels, vocab, tokenizer, max_seq_len=128):
        """
        Initialize the dataset.

        Args:
            texts: List of text strings
            labels: List of label integers (0, 1 for binary classification)
            vocab: Vocabulary mapping tokens to indices
            tokenizer: Function to tokenize text into tokens
            max_seq_len: Maximum sequence length (longer sequences are truncated)
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Special tokens - these are standard in most NLP tasks
        self.pad_idx = vocab["<pad>"]  # Index for padding
        self.bos_idx = vocab["<bos>"]  # Beginning of sequence
        self.eos_idx = vocab["<eos>"]  # End of sequence
        self.unk_idx = vocab["<unk>"]  # Unknown token

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        This method:
        1. Gets the text and its label
        2. Tokenizes the text
        3. Converts tokens to numerical indices using the vocabulary
        4. Truncates or pads the sequence to max_seq_len
        5. Returns the tensor and label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Step 1: Tokenize the text
        # The tokenizer splits text into tokens (words, subwords, or characters)
        # tokens: List[str] of length (num_tokens)
        tokens = self.tokenizer(text)

        # Step 2: Convert tokens to indices
        # vocab(token) looks up the index for each token
        # We add special tokens: <bos> at start, <eos> at end
        # token_indices: List[int] of length (num_tokens + 2) = (seq_len_with_special)
        token_indices = [self.bos_idx] + self.vocab(tokens) + [self.eos_idx]

        # Step 3: Truncate if sequence is too long
        if len(token_indices) > self.max_seq_len:
            token_indices = token_indices[: self.max_seq_len]  # Shape: (max_seq_len,)

        # Step 4: Create attention mask
        # The mask tells the transformer which positions are real tokens vs padding
        # 1 = real token, 0 = padding
        # attention_mask: List[int] of length (seq_len_after_truncate)
        attention_mask = [1] * len(token_indices)

        # Step 5: Pad the sequence to max_seq_len
        # We pad with self.pad_idx to make all sequences the same length
        padding_length = self.max_seq_len - len(token_indices)
        # token_indices: List[int] of length (max_seq_len)
        token_indices = token_indices + [self.pad_idx] * padding_length
        # attention_mask: List[int] of length (max_seq_len)
        attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        # LongTensor is used for integer indices
        # token_tensor: torch.Tensor of shape (max_seq_len,) dtype=torch.long
        token_tensor = torch.tensor(token_indices, dtype=torch.long)
        # mask_tensor: torch.Tensor of shape (max_seq_len,) dtype=torch.long
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        # label_tensor: torch.Tensor of shape () (scalar) dtype=torch.long
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": token_tensor,  # Shape: (max_seq_len,)
            "attention_mask": mask_tensor,  # Shape: (max_seq_len,)
            "label": label_tensor,  # Shape: ()
        }


def create_synthetic_data(num_samples=1000):
    """
    Create synthetic text classification data for demonstration.

    In a real project, you would load data from files or datasets.
    We create simple positive/negative sentences here for demonstration.
    """
    positive_words = [
        "happy",
        "joy",
        "love",
        "great",
        "excellent",
        "wonderful",
        "amazing",
    ]
    negative_words = ["sad", "hate", "terrible", "awful", "bad", "worst", "horrible"]

    texts = []
    labels = []

    for _ in range(num_samples // 2):
        # Create positive examples
        words = np.random.choice(positive_words, size=np.random.randint(3, 10))
        text = " ".join(words) + " movie"
        texts.append(text)
        labels.append(1)  # Positive label

        # Create negative examples
        words = np.random.choice(negative_words, size=np.random.randint(3, 10))
        text = " ".join(words) + " movie"
        texts.append(text)
        labels.append(0)  # Negative label

    return texts, labels


def build_vocabulary(texts, tokenizer, special_tokens=None):
    """
    Build a vocabulary from a list of texts.

    The vocabulary maps:
    - Tokens (strings) to indices (integers)
    - Indices to tokens

    This is essential for converting text to numerical representations
    that the model can process.

    Returns:
        vocab: Vocab object where vocab[token] -> index (int)
               and vocab.get_itos()[index] -> token (str)
    """

    # Default specials are built inside the function: a mutable default
    # argument would be shared across every call.
    if special_tokens is None:
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Tokenize all texts and flatten into one list
    def yield_tokens():
        for text in texts:
            yield tokenizer(text)

    # Build vocabulary with special tokens
    # vocab: Vocab object that maps str -> int and int -> str
    vocab = build_vocab_from_iterator(
        yield_tokens(),
        min_freq=1,  # Include all tokens that appear at least once
        specials=special_tokens,
    )

    # Set the default index for unknown tokens
    vocab.set_default_index(vocab["<unk>"])

    return vocab


# =============================================================================
# PART 2: MODEL DEFINITION
# =============================================================================


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer Models.

    Transformers don't have inherent notion of sequence order (they're
    permutation-equivariant). Positional encoding adds information about
    the position of each token in the sequence.

    This implementation uses sinusoidal positional encodings as described
    in "Attention Is All You Need" (Vaswani et al., 2017).

    The formula is:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos: position in the sequence
    - i: dimension index
    - d_model: embedding dimension

    Input:  x of shape (batch_size, seq_len, d_model)
    Output: x + positional_encoding of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            dropout: Dropout rate for positional encodings
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        # position: torch.Tensor of shape (max_len,)  # [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)

        # div_term: torch.Tensor of shape (d_model/2,)  # Exponential terms for different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe: torch.Tensor of shape (max_len, 1, d_model)  # Positional encoding matrix
        pe = torch.zeros(max_len, 1, d_model)
        # Fill even dimensions (0, 2, 4, ...) with sin
        # pe[:, 0, 0::2]: shape (max_len, d_model/2)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        # Fill odd dimensions (1, 3, 5, ...) with cos
        # pe[:, 0, 1::2]: shape (max_len, d_model/2)
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos

        # Register as buffer (not a learnable parameter, but saved with model)
        # pe: torch.Tensor of shape (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added, shape (batch_size, seq_len, d_model)
        """
        # x: torch.Tensor of shape (batch_size, seq_len, d_model)
        # self.pe: torch.Tensor of shape (max_len, 1, d_model)

        # Select only the first seq_len positions from pe
        # self.pe[:x.size(1)]: torch.Tensor of shape (seq_len, 1, d_model)
        # Broadcasting: (batch_size, seq_len, d_model) + (seq_len, 1, d_model) -> (batch_size, seq_len, d_model)
        x = x + self.pe[: x.size(1)]  # Take only as many positions as needed

        # Apply dropout
        # x: torch.Tensor of shape (batch_size, seq_len, d_model)
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """
    Transformer-based Encoder Model for text classification.

    This is a complete transformer encoder model that:
    1. Embeds input tokens into a continuous vector space
    2. Adds positional information
    3. Processes through multiple transformer encoder layers
    4. Pools the sequence representations
    5. Makes a classification prediction

    Architecture:
    Input -> Token Embedding -> Positional Encoding -> Transformer Encoder -> Pooling -> Classifier

    MODEL SHAPES:
    -------------
    Forward pass flow for a batch of B samples with sequence length S:
      input_ids:      (B, S)         -> token_embedding -> (B, S, d_model)
                                          -> + positional encoding -> (B, S, d_model)
                                          -> transformer_encoder -> (B, S, d_model)
                                          -> permute to (B, d_model, S)
                                          -> pooling -> (B, d_model, 1)
                                          -> squeeze -> (B, d_model)
                                          -> classifier -> (B, num_classes)

    Where:
      B = batch_size
      S = sequence_length (<= max_seq_len)
      d_model = embedding dimension
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=2,
    ):
        """
        Initialize the transformer encoder model.

        Args:
            vocab_size: Number of tokens in vocabulary
            d_model: Dimension of token embeddings and model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()

        # Store configuration
        self.d_model = d_model
        self.nhead = nhead

        # 1. Token Embedding Layer
        # This converts token indices to continuous vectors
        # self.token_embedding: nn.Embedding with weight of shape (vocab_size, d_model)
        # Input: (batch_size, seq_len) of int indices -> Output: (batch_size, seq_len, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        # Adds information about token positions
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # 3. Transformer Encoder
        # The core transformer architecture
        # Each EncoderLayer contains:
        #   - Multi-head self-attention
        #   - Position-wise feedforward network
        #   - Layer normalization
        #   - Residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Input dimension for each token
            nhead=nhead,  # Number of parallel attention heads
            dim_feedforward=dim_feedforward,  # Dimension of FFN hidden layer
            dropout=dropout,  # Dropout probability
            activation="gelu",  # GELU activation (common in transformers)
            batch_first=True,  # Input shape: (batch, seq_len, d_model) vs (seq_len, batch, d_model)
        )

        # Stack multiple encoder layers
        # self.transformer_encoder: nn.TransformerEncoder
        # Input: (batch, seq_len, d_model) -> Output: (batch, seq_len, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,  # Number of encoder layers to stack
        )

        # 4. Pooling Layer
        # We use mean pooling over the sequence dimension
        # This converts (batch, seq_len, d_model) -> (batch, d_model)
        # Alternative: use the [CLS] token or max pooling
        # self.pooling: nn.AdaptiveAvgPool1d
        # Input: (batch, d_model, seq_len) -> Output: (batch, d_model, 1)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # 5. Classifier Head
        # Maps the pooled representation to class logits
        # self.classifier: nn.Linear
        # Input: (batch, d_model) -> Output: (batch, num_classes)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.

        Proper weight initialization is important for training stability.
        We use Xavier uniform initialization for embeddings and linear layers.
        """
        init_range = 0.1  # Small range for stable training

        # Initialize token embeddings
        # self.token_embedding.weight: torch.Tensor of shape (vocab_size, d_model)
        nn.init.uniform_(self.token_embedding.weight, -init_range, init_range)

        # Initialize classifier weights
        # self.classifier.weight: torch.Tensor of shape (num_classes, d_model)
        # self.classifier.bias: torch.Tensor of shape (num_classes,)
        nn.init.uniform_(self.classifier.weight, -init_range, init_range)
        nn.init.zeros_(self.classifier.bias)  # Zero bias is common

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token indices, shape (batch_size, seq_len)
            attention_mask: Tensor of attention mask, shape (batch_size, seq_len)
                           1 for real tokens, 0 for padding

        Returns:
            logits: Output logits for each class, shape (batch_size, num_classes)
        """
        # Step 1: Embed tokens
        # input_ids: torch.Tensor of shape (batch_size, seq_len)
        # self.token_embedding(input_ids): torch.Tensor of shape (batch_size, seq_len, d_model)
        embedded = self.token_embedding(input_ids)  # Shape: (B, S, d_model)

        # Step 2: Add positional encoding
        # Scale embedding by sqrt(d_model) as in original transformer paper
        # embedded: torch.Tensor of shape (B, S, d_model)
        embedded = embedded * math.sqrt(self.d_model)
        # self.pos_encoder(embedded): torch.Tensor of shape (B, S, d_model)
        embedded = self.pos_encoder(embedded)

        # Step 3: Prepare for transformer
        # Transformer expects (batch, seq_len, d_model)
        # We already have this shape: embedded: (B, S, d_model)

        # Step 4: Create attention mask for transformer
        # The transformer needs a mask to ignore padding tokens
        # attention_mask: torch.Tensor of shape (batch_size, seq_len)
        if attention_mask is not None:
            # Convert mask to boolean and invert
            # For nn.TransformerEncoder, we need src_key_padding_mask:
            #   torch.Tensor of shape (batch, src_seq_len) with True for padded positions
            # attention_mask has 1 for real tokens, 0 for padding
            # (attention_mask == 0): torch.Tensor of shape (B, S) with True where padded
            src_key_padding_mask = attention_mask == 0  # Shape: (B, S), True where padded
        else:
            src_key_padding_mask = None

        # Step 5: Pass through transformer encoder
        # transformer_encoder expects:
        #   - src: torch.Tensor of shape (batch, seq_len, d_model)
        #   - src_key_padding_mask: torch.Tensor of shape (batch, seq_len) or None
        # Returns: torch.Tensor of shape (batch, seq_len, d_model)
        transformer_output = self.transformer_encoder(
            src=embedded,  # Shape: (B, S, d_model)
            src_key_padding_mask=src_key_padding_mask,  # Shape: (B, S)
        )  # Output: (B, S, d_model)

        # Step 6: Pool the sequence representations
        # We use global average pooling
        # transformer_output: torch.Tensor of shape (B, S, d_model)
        # We want: torch.Tensor of shape (B, d_model)

        # Permute for pooling: (batch, d_model, seq_len)
        # .permute(0, 2, 1) swaps dim 1 and dim 2
        # pooled: torch.Tensor of shape (B, d_model, S)
        pooled = transformer_output.permute(0, 2, 1)

        # Apply average pooling across sequence dimension (dim=2)
        # self.pooling(pooled): torch.Tensor of shape (B, d_model, 1)
        pooled = self.pooling(pooled)

        # Squeeze to remove last dimension (dim=2)
        # pooled: torch.Tensor of shape (B, d_model)
        pooled = pooled.squeeze(-1)

        # Step 7: Classification
        # pooled: torch.Tensor of shape (B, d_model)
        # self.classifier(pooled): torch.Tensor of shape (B, num_classes)
        logits = self.classifier(pooled)  # Shape: (B, num_classes)

        # logits: torch.Tensor of shape (batch_size, num_classes)
        return logits


# =============================================================================
# PART 3: TRAINING SETUP
# =============================================================================


def collate_batch(batch):
    """
    Collate function for DataLoader.

    The DataLoader passes batches to the model. Each batch is a list
    of samples from the dataset. This function combines them into tensors.

    Args:
        batch: List of samples, where each sample is a dict with:
               - input_ids: token indices, shape (max_seq_len,)
               - attention_mask: attention mask, shape (max_seq_len,)
               - label: classification label, shape ()

    Returns:
        Dict with batched tensors:
          - input_ids: shape (batch_size, max_seq_len)
          - attention_mask: shape (batch_size, max_seq_len)
          - labels: shape (batch_size,)
    """
    # Extract individual items from batch
    # Each item['input_ids']: torch.Tensor of shape (max_seq_len,)
    input_ids = [item["input_ids"] for item in batch]
    # Each item['attention_mask']: torch.Tensor of shape (max_seq_len,)
    attention_masks = [item["attention_mask"] for item in batch]
    # Each item['label']: torch.Tensor of shape ()
    labels = [item["label"] for item in batch]

    # Stack tensors along batch dimension (dim=0)
    # batched_input_ids: torch.Tensor of shape (batch_size, max_seq_len)
    batched_input_ids = torch.stack(input_ids, dim=0)
    # batched_attention_masks: torch.Tensor of shape (batch_size, max_seq_len)
    batched_attention_masks = torch.stack(attention_masks, dim=0)
    # batched_labels: torch.Tensor of shape (batch_size,)
    batched_labels = torch.stack(labels, dim=0)

    return {
        "input_ids": batched_input_ids,  # Shape: (batch_size, max_seq_len)
        "attention_mask": batched_attention_masks,  # Shape: (batch_size, max_seq_len)
        "labels": batched_labels,  # Shape: (batch_size,)
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, patience=3):
    """
    Train the transformer model.

    This function implements the complete training loop including:
    - Forward pass
    - Loss calculation
    - Backward pass (gradient computation)
    - Optimization step
    - Validation
    - Early stopping

    Args:
        model: The transformer model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimization algorithm
        criterion: Loss function
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping

    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(DEVICE)

    # Track best validation accuracy for early stopping
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Training history for plotting/analysis
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Set model to training mode
        # This affects dropout and batch normalization layers
        model.train()

        # Track epoch statistics
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        # ===== TRAINING LOOP =====
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            # batch['input_ids']: torch.Tensor of shape (batch_size, max_seq_len)
            input_ids = batch["input_ids"].to(DEVICE)
            # batch['attention_mask']: torch.Tensor of shape (batch_size, max_seq_len)
            attention_mask = batch["attention_mask"].to(DEVICE)
            # batch['labels']: torch.Tensor of shape (batch_size,)
            labels = batch["labels"].to(DEVICE)

            # Zero gradients from previous batch
            # PyTorch accumulates gradients by default
            optimizer.zero_grad()

            # Forward pass
            # This computes the model's predictions
            # model(input_ids, attention_mask): torch.Tensor of shape (batch_size, num_classes)
            logits = model(input_ids, attention_mask)

            # Calculate loss
            # CrossEntropyLoss expects:
            #   - logits: (batch_size, num_classes)
            #   - labels: (batch_size,) with class indices
            # loss: torch.Tensor of shape () (scalar)
            loss = criterion(logits, labels)

            # Backward pass
            # This computes gradients of the loss w.r.t. all model parameters
            # After this, all model parameters have .grad attribute
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            # This is important for transformer models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            # This updates the model parameters
            optimizer.step()

            # Track statistics
            # loss.item(): float, the actual loss value
            total_train_loss += loss.item() * input_ids.size(0)  # Multiply by batch size

            # Calculate accuracy
            # torch.max(logits, dim=1): Returns (values, indices) along dim 1
            # predicted: torch.Tensor of shape (batch_size,) with predicted class indices
            _, predicted = torch.max(logits, dim=1)
            # (predicted == labels): torch.Tensor of shape (batch_size,) with True/False
            # .sum().item(): int, number of correct predictions
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += input_ids.size(0)  # Add batch size

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_loss = loss.item()
                batch_acc = (predicted == labels).sum().item() / input_ids.size(0)
                print(f"  Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {batch_loss:.4f}, Acc = {batch_acc:.4f}")

        # Calculate epoch statistics
        epoch_train_loss = total_train_loss / total_train_samples  # float
        epoch_train_acc = total_train_correct / total_train_samples  # float

        # ===== VALIDATION LOOP =====
        model.eval()  # Set to evaluation mode (disables dropout)

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        # Disable gradient computation for validation (faster and less memory)
        with torch.no_grad():
            for batch in val_loader:
                # Each batch has same structure as training
                input_ids = batch["input_ids"].to(DEVICE)  # Shape: (batch_size, max_seq_len)
                attention_mask = batch["attention_mask"].to(DEVICE)  # Shape: (batch_size, max_seq_len)
                labels = batch["labels"].to(DEVICE)  # Shape: (batch_size,)

                # Forward pass
                # logits: torch.Tensor of shape (batch_size, num_classes)
                logits = model(input_ids, attention_mask)

                # Calculate loss
                # loss: torch.Tensor of shape () (scalar)
                loss = criterion(logits, labels)

                # Track statistics
                total_val_loss += loss.item() * input_ids.size(0)

                # Calculate accuracy
                # predicted: torch.Tensor of shape (batch_size,)
                _, predicted = torch.max(logits, dim=1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += input_ids.size(0)

        epoch_val_loss = total_val_loss / total_val_samples  # float
        epoch_val_acc = total_val_correct / total_val_samples  # float

        # Store history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train: Loss = {epoch_train_loss:.4f}, Acc = {epoch_train_acc:.4f}")
        print(f"  Val:   Loss = {epoch_val_loss:.4f}, Acc = {epoch_val_acc:.4f}")
        print(f"  Time:  {epoch_time:.2f}s")

        # Early stopping check
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            epochs_without_improvement = 0

            # Save best model
            # model.state_dict(): OrderedDict with all model parameters
            # Each parameter is a torch.Tensor with appropriate shape
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  New best model saved with val_acc = {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Load best model weights
    model.load_state_dict(torch.load("best_model.pth"))

    return model, history


# =============================================================================
# PART 4: LIVE INFERENCE
# =============================================================================


def predict(model, text, tokenizer, vocab, max_seq_len=128, device=DEVICE):
    """
    Run inference on a single text sample.

    This function shows how to:
    1. Tokenize new text
    2. Convert to numerical representation
    3. Run through the model
    4. Get the prediction

    Args:
        model: Trained transformer model
        text: Input text string
        tokenizer: Tokenizer function
        vocab: Vocabulary
        max_seq_len: Maximum sequence length
        device: Device to run inference on

    Returns:
        Predicted class index and probability
    """
    # Set model to evaluation mode
    model.eval()

    # Tokenize and prepare the text (same as in dataset)
    tokens = tokenizer(text)  # List[str] of length (num_tokens)
    vocab.set_default_index(vocab["<unk>"])
    # token_indices: List[int] of length (num_tokens + 2)
    token_indices = [vocab["<bos>"]] + vocab(tokens) + [vocab["<eos>"]]

    # Truncate if needed
    if len(token_indices) > max_seq_len:
        token_indices = token_indices[:max_seq_len]  # List[int] of length (max_seq_len)

    # Create attention mask
    # attention_mask: List[int] of length (seq_len)
    attention_mask = [1] * len(token_indices)

    # Pad
    padding_length = max_seq_len - len(token_indices)
    # token_indices: List[int] of length (max_seq_len)
    token_indices = token_indices + [vocab["<pad>"]] * padding_length
    # attention_mask: List[int] of length (max_seq_len)
    attention_mask = attention_mask + [0] * padding_length

    # Convert to tensors and add batch dimension
    # input_tensor: torch.Tensor of shape (1, max_seq_len) - batch of 1 sample
    input_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
    # mask_tensor: torch.Tensor of shape (1, max_seq_len) - batch of 1 sample
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)

    # Move to device
    input_tensor = input_tensor.to(device)  # Shape: (1, max_seq_len)
    mask_tensor = mask_tensor.to(device)  # Shape: (1, max_seq_len)

    # Run inference (no gradient computation)
    with torch.no_grad():
        # model(input_tensor, mask_tensor): torch.Tensor of shape (1, num_classes)
        logits = model(input_tensor, mask_tensor)

        # Get probabilities using softmax
        # F.softmax(logits, dim=1): torch.Tensor of shape (1, num_classes)
        probs = F.softmax(logits, dim=1)

        # Get predicted class
        # torch.max(logits, dim=1): Returns (values, indices) along dim 1
        # predicted_class: torch.Tensor of shape (1,) -> .item() gives int
        _, predicted_class = torch.max(logits, dim=1)
        predicted_class = predicted_class.item()  # int
        confidence = probs[0, predicted_class].item()  # float

    return predicted_class, confidence


def interactive_inference(model, tokenizer, vocab, max_seq_len=128):
    """
    Run interactive inference allowing user to input text.

    This demonstrates live inference with a trained model.
    """
    print("\n" + "=" * 60)
    print("LIVE INFERENCE MODE")
    print("=" * 60)
    print("Enter text to classify (or 'quit' to exit):")

    while True:
        text = input("\n> ").strip()

        if text.lower() == "quit":
            break

        if not text:
            print("Please enter some text.")
            continue

        # Run prediction
        start_time = time.time()
        predicted_class, confidence = predict(model, text, tokenizer, vocab, max_seq_len, DEVICE)
        inference_time = time.time() - start_time

        # Display results
        class_name = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
        print(f"Prediction: {class_name} (class {predicted_class})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference time: {inference_time * 1000:.2f}ms")


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================


def main():
    """
    Main function that ties everything together.

    This function:
    1. Creates synthetic data
    2. Builds vocabulary and tokenizer
    3. Creates datasets and dataloaders
    4. Initializes the model
    5. Trains the model
    6. Runs live inference
    """
    print("PyTorch Transformer Encoder Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Create and prepare data
    # -------------------------------------------------------------------------
    print("\nStep 1: Creating and preparing data...")

    # Create synthetic data
    num_samples = 1000
    texts, labels = create_synthetic_data(num_samples)

    # Split into train and validation
    split_idx = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    val_texts, val_labels = texts[split_idx:], labels[split_idx:]

    print(f"  Total samples: {len(texts)}")
    print(f"  Training samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")

    # Create tokenizer
    # This is a simple whitespace tokenizer
    # In production, you'd use subword tokenizers like BPE or WordPiece
    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary
    vocab = build_vocabulary(train_texts, tokenizer)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Sample tokens: {list(vocab.get_itos())[:20]}")

    # Create datasets
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        tokenizer=tokenizer,
        max_seq_len=128,
    )

    val_dataset = TextClassificationDataset(
        texts=val_texts,
        labels=val_labels,
        vocab=vocab,
        tokenizer=tokenizer,
        max_seq_len=128,
    )

    # Create dataloaders
    # The DataLoader handles batching, shuffling, and parallel loading
    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        collate_fn=collate_batch,
        num_workers=0,  # 0 for simplicity; use >0 for parallel loading
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        collate_fn=collate_batch,
        num_workers=0,
    )

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # STEP 2: Initialize model
    # -------------------------------------------------------------------------
    print("\nStep 2: Initializing model...")

    # Model hyperparameters
    vocab_size = len(vocab)
    d_model = 128  # Embedding dimension
    nhead = 4  # Number of attention heads
    num_layers = 2  # Number of transformer layers
    num_classes = 2  # Binary classification

    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    print("  Model architecture:")
    print(f"    - Embedding dim: {d_model}")
    print(f"    - Attention heads: {nhead}")
    print(f"    - Transformer layers: {num_layers}")

    # -------------------------------------------------------------------------
    # STEP 3: Setup training
    # -------------------------------------------------------------------------
    print("\nStep 3: Setting up training...")

    # Loss function
    # CrossEntropyLoss is standard for classification
    # It combines log_softmax and NLLLoss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # AdamW is commonly used for transformer models
    # It includes weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Learning rate
        weight_decay=0.01,  # L2 regularization
    )

    # Learning rate scheduler (optional)
    # This reduces the learning rate during training
    # Common for fine-tuning but less critical for small examples
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # noqa: F841  (demo setup; unused in this minimal loop)
        optimizer,
        mode="max",  # We're tracking accuracy (higher is better)
        factor=0.1,  # Reduce learning rate by this factor
        patience=2,  # Wait this many epochs without improvement
        verbose=True,
    )

    print("  Optimizer: AdamW")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print("  Loss function: CrossEntropyLoss")

    # -------------------------------------------------------------------------
    # STEP 4: Train the model
    # -------------------------------------------------------------------------
    print("\nStep 4: Training model...")

    num_epochs = 10
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        patience=3,
    )

    print("\nTraining complete!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")

    # -------------------------------------------------------------------------
    # STEP 5: Run live inference
    # -------------------------------------------------------------------------
    print("\nStep 5: Running live inference...")

    # Run interactive inference
    interactive_inference(model=model, tokenizer=tokenizer, vocab=vocab, max_seq_len=128)

    print("\nDone!")


if __name__ == "__main__":
    main()
