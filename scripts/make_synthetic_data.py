from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


LEVEL_WORDS = {
    0: "cat dog food school simple hello",
    1: "travel weather family weekend because",
    2: "technology environment opinion although context",
    3: "economics policy contrast implication sophisticated",
    4: "epistemology rhetoric nuance counterfactual interdisciplinary",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(args.n):
        label = i % 5
        a = np.full((args.image_size, args.image_size, 3), 40 + label * 35, dtype=np.uint8)
        a += rng.integers(0, 20, size=a.shape, dtype=np.uint8)
        img = Image.fromarray(a)
        draw = ImageDraw.Draw(img)
        draw.rectangle([15 + label * 5, 15, 80 + label * 10, 80], outline=(255, 255, 255), width=3)
        name = f"img_{i:04d}.png"
        img.save(img_dir / name)
        success_p = 0.85 - 0.12 * label
        r = rng.random()
        outcome = 1 if r < success_p else (2 if r > 0.95 else 0)
        text = LEVEL_WORDS[label] + f" sample {i}"
        rows.append({
            "image_path": name,
            "text": text,
            "label": label,
            "sequence_id": f"learner_{i // 20:02d}",
            "timestep": i % 20,
            "outcome": outcome,
        })
    pd.DataFrame(rows).to_csv(out / "train.csv", index=False)
    print("wrote", out / "train.csv", "and", img_dir)


if __name__ == "__main__":
    main()
