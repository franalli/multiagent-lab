"""
finetune.py

Supervised fine-tuning of a Mistral base model with LoRA on a small
instruction dataset. Trains the adapter and reports validation loss.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "mistralai/Mistral-7B-v0.1"
EPOCHS = 3
LR = 1e-3
BATCH = 8
MAX_LEN = 512


class SFTDataset(Dataset):
    def __init__(self, path):
        self.rows = json.load(open(path))
        self.tok = AutoTokenizer.from_pretrained(MODEL)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        text = row["prompt"] + row["completion"]
        enc = self.tok(text, max_length=MAX_LEN, padding="max_length")
        ids = torch.tensor(enc["input_ids"])
        return {"input_ids": ids, "labels": ids}


def get_loaders(path):
    ds = SFTDataset(path)
    n = len(ds)
    train = Subset(ds, range(0, int(n * 0.9)))
    val = Subset(ds, range(int(n * 0.9), n))
    return DataLoader(train, batch_size=BATCH), DataLoader(val, batch_size=BATCH)


def evaluate(model, loader):
    total = 0
    for batch in loader:
        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        out = model(input_ids=ids, labels=labels)
        total += out.loss.item()
    print(total / len(loader))


def train(path):
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    config = LoraConfig(r=8, target_modules=["lm_head"])
    model = get_peft_model(model, config)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader, val_loader = get_loaders(path)

    losses = []
    for epoch in range(EPOCHS):
        for batch in train_loader:
            ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            out = model(input_ids=ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            losses.append(loss)
        evaluate(model, val_loader)

    torch.save(model.state_dict(), "ckpt.pt")
