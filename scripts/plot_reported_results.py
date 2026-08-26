from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data_paper_reported_results.csv")
    p.add_argument("--out", default="outputs/reported_accuracy.png")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    pivot = df.pivot(index="method", columns="dataset", values="accuracy")
    order = ["BERT-base", "VGG-LSTM", "ResNet-SVM", "DualPath-ATT", "SocialEdu", "SMELL"]
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.set_title("Paper-reported SMELL benchmark accuracy")
    ax.set_ylim(60, 90)
    ax.legend(title="Dataset")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
