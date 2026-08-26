from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help=".npz with logits and labels")
    args = p.parse_args()
    arr = np.load(args.predictions)
    logits = torch.tensor(arr["logits"])
    labels = arr["labels"]
    pred = logits.argmax(-1).numpy()
    print(f"Accuracy: {accuracy_score(labels, pred):.4f}")
    print(f"Macro-F1: {f1_score(labels, pred, average='macro'):.4f}")
    print("BLEU/RCP/SPG require generated exercise references or the offline interaction simulator and are not inferred from class logits alone.")


if __name__ == "__main__":
    main()
