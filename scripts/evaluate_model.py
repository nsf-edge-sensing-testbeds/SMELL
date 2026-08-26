from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from src.data.dataset import SocialMediaELLDataset
from src.data.vocab import Vocabulary
from src.models.smell import SMELLModel
from src.utils.config import load_config, resolve_device


def restore_vocab(ckpt):
    v = Vocabulary(max_size=max(len(ckpt["vocab_itos"]), 3))
    v.stoi = ckpt["vocab_stoi"]
    v.itos = ckpt["vocab_itos"]
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/smell_best.pt")
    p.add_argument("--csv", default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--save-predictions", default="outputs/predictions.npz")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["train"]["device"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = restore_vocab(ckpt)
    dc, mc = cfg["data"], cfg["model"]
    ds = SocialMediaELLDataset(
        args.csv or dc["csv_path"], args.image_root or dc["image_root"], vocab,
        image_size=dc["image_size"], max_length=dc["max_length"], train=False,
    )
    loader = DataLoader(ds, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])
    model = SMELLModel(len(vocab), mc["num_classes"], mc["text_dim"], mc["fusion_dim"], mc["text_layers"], mc["num_heads"], mc["ff_dim"], mc["dropout"], False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device))
            logits.append(out["logits"].cpu())
            labels.append(batch["label"].cpu())
    logits = torch.cat(logits).numpy()
    labels = torch.cat(labels).numpy()
    pred = logits.argmax(axis=1)
    print(f"Accuracy : {accuracy_score(labels, pred):.4f}")
    print(f"Macro-F1 : {f1_score(labels, pred, average='macro'):.4f}")
    print(classification_report(labels, pred, digits=4, zero_division=0))
    out = _Path(args.save_predictions)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=logits, labels=labels)
    print("saved", out)


if __name__ == "__main__":
    main()
