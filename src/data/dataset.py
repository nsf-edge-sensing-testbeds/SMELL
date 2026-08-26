from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .vocab import Vocabulary


LABEL_MAP = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4}
OUTCOME_MAP = {"incorrect": 0, "correct": 1, "skipped": 2, 0: 0, 1: 1, 2: 2}


class SocialMediaELLDataset(Dataset):
    """CSV columns required: image_path,text,label.

    Optional columns for offline learner sequences:
    sequence_id,timestep,outcome.
    label may be integer 0..4 or CEFR string A1..C1.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        vocab: Vocabulary,
        image_size: int = 224,
        max_length: int = 64,
        train: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.vocab = vocab
        self.max_length = max_length
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.df.iloc[idx]
        img = Image.open(self.image_root / row["image_path"]).convert("RGB")
        image = self.transform(img)
        ids, mask = self.vocab.encode(str(row["text"]), self.max_length)
        raw_label = row["label"]
        label = LABEL_MAP.get(raw_label, int(raw_label) if str(raw_label).isdigit() else 0)
        item = {
            "image": image,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "text": str(row["text"]),
            "index": int(idx),
        }
        if "outcome" in row.index:
            raw = row["outcome"]
            outcome = OUTCOME_MAP.get(raw, int(raw) if str(raw).isdigit() else 0)
            item["outcome"] = torch.tensor(outcome, dtype=torch.long)
        if "sequence_id" in row.index:
            item["sequence_id"] = str(row["sequence_id"])
        if "timestep" in row.index:
            item["timestep"] = int(row["timestep"])
        return item


def build_vocab_from_csv(csv_path: str | Path, max_size: int = 30000, min_freq: int = 1) -> Vocabulary:
    df = pd.read_csv(csv_path)
    vocab = Vocabulary(max_size=max_size, min_freq=min_freq)
    vocab.build(df["text"].astype(str).tolist())
    return vocab
