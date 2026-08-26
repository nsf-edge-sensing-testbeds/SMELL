from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import torch

from src.models.proficiency_hmm import ProficiencyHMM
from src.utils.config import load_config, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--embeddings", default="outputs/embeddings.npz")
    p.add_argument("--csv", default=None, help="CSV must contain outcome; sequence_id/timestep recommended")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--output", default="outputs/proficiency_hmm.pt")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["train"]["device"])
    arr = np.load(args.embeddings)
    z = torch.tensor(arr["embedding"], dtype=torch.float32)
    df = pd.read_csv(args.csv or cfg["data"]["csv_path"])
    if "outcome" not in df:
        raise ValueError("CSV needs an 'outcome' column encoded as incorrect/correct/skipped or 0/1/2")
    mapping = {"incorrect": 0, "correct": 1, "skipped": 2}
    obs_np = np.array([mapping.get(str(v), int(v) if str(v).isdigit() else 0) for v in df["outcome"]], dtype=np.int64)

    hc = cfg["hmm"]
    hmm = ProficiencyHMM(
        z_dim=cfg["model"]["fusion_dim"], num_states=hc["num_states"], state_dim=hc["state_dim"],
        state_difficulties=hc["state_difficulties"], exploration_sigma=hc["exploration_sigma"],
    ).to(device)
    opt = torch.optim.Adam(hmm.parameters(), lr=1e-3)

    if "sequence_id" in df:
        groups = [g.sort_values("timestep") if "timestep" in g else g for _, g in df.groupby("sequence_id")]
    else:
        groups = [df]

    for epoch in range(args.epochs):
        total = 0.0
        for g in groups:
            idx = torch.tensor(g.index.to_numpy(), dtype=torch.long)
            zg = z[idx].unsqueeze(0).to(device)
            og = torch.tensor(obs_np[g.index.to_numpy()], dtype=torch.long).unsqueeze(0).to(device)
            opt.zero_grad()
            loss = hmm.sequence_nll(zg, og)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch={epoch+1:03d} hmm_nll={total / len(groups):.4f}")

    # Paper states Baum-Welch updates for A; apply one transition-only EM update per sequence.
    for g in groups:
        idx_np = g.index.to_numpy()
        hmm.baum_welch_transition_update(z[idx_np].to(device), torch.tensor(obs_np[idx_np], dtype=torch.long, device=device), concentration=hc["transition_prior"])
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": hmm.state_dict(), "config": cfg}, args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
