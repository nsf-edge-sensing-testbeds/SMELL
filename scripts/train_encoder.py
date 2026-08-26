from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SocialMediaELLDataset, build_vocab_from_csv
from src.models.smell import SMELLModel
from src.train_utils import train_one_epoch, evaluate_classifier, save_checkpoint
from src.utils.config import load_config, seed_everything, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    dc = cfg["data"]
    vocab = build_vocab_from_csv(dc["csv_path"], cfg["model"]["vocab_size"], dc["min_freq"])
    train_full = SocialMediaELLDataset(
        dc["csv_path"], dc["image_root"], vocab,
        image_size=dc["image_size"], max_length=dc["max_length"], train=True,
    )
    val_full = SocialMediaELLDataset(
        dc["csv_path"], dc["image_root"], vocab,
        image_size=dc["image_size"], max_length=dc["max_length"], train=False,
    )
    n = len(train_full)
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set, val_set = Subset(train_full, train_idx), Subset(val_full, val_idx)
    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, num_workers=dc["num_workers"])
    val_loader = DataLoader(val_set, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])

    mc = cfg["model"]
    model = SMELLModel(
        vocab_size=len(vocab), num_classes=mc["num_classes"], text_dim=mc["text_dim"],
        fusion_dim=mc["fusion_dim"], text_layers=mc["text_layers"], num_heads=mc["num_heads"],
        ff_dim=mc["ff_dim"], dropout=mc["dropout"], visual_pretrained=mc["visual_pretrained"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    best = -1.0
    out_dir = Path(cfg["train"]["checkpoint_dir"])
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = train_one_epoch(model, train_loader, opt, device)
        va = evaluate_classifier(model, val_loader, device)
        sched.step()
        print(f"epoch={epoch:03d} train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.4f} val_acc={va['accuracy']:.4f}")
        if va["accuracy"] > best:
            best = va["accuracy"]
            save_checkpoint(out_dir / "smell_best.pt", model, vocab, {"val_accuracy": best, "config": cfg})


if __name__ == "__main__":
    main()
