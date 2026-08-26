from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(image, ids, mask)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * labels.size(0)
        total_correct += int((out["logits"].argmax(-1) == labels).sum())
        total += labels.size(0)
    return {"loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    logits_all, labels_all = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        out = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        labels = batch["label"].to(device)
        total_correct += int((out["logits"].argmax(-1) == labels).sum())
        total += labels.size(0)
        logits_all.append(out["logits"].cpu())
        labels_all.append(labels.cpu())
    return {
        "accuracy": total_correct / max(total, 1),
        "logits": torch.cat(logits_all) if logits_all else torch.empty(0),
        "labels": torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long),
    }


def save_checkpoint(path: str | Path, model, vocab, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "vocab_stoi": vocab.stoi,
        "vocab_itos": vocab.itos,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
