from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
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
    p.add_argument("--output", default="outputs/embeddings.npz")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["train"]["device"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = restore_vocab(ckpt)
    dc, mc = cfg["data"], cfg["model"]
    ds = SocialMediaELLDataset(dc["csv_path"], dc["image_root"], vocab, dc["image_size"], dc["max_length"], train=False)
    loader = DataLoader(ds, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])
    model = SMELLModel(len(vocab), mc["num_classes"], mc["text_dim"], mc["fusion_dim"], mc["text_layers"], mc["num_heads"], mc["ff_dim"], mc["dropout"], False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    zs, dscores, labels, indices = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device))
            zs.append(out["embedding"].cpu().numpy())
            dscores.append(out["difficulty"].cpu().numpy())
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embedding=np.concatenate(zs), difficulty=np.concatenate(dscores), label=np.concatenate(labels), index=np.concatenate(indices))
    print("saved", args.output)


if __name__ == "__main__":
    main()
