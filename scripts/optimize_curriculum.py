from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import numpy as np

from src.curriculum import simulated_proficiency_gain
from src.optim.afoa import AFOA, AFOAConfig
from src.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--embeddings", default="outputs/embeddings.npz")
    p.add_argument("--output", default="outputs/afoa_result.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    difficulties = np.load(args.embeddings)["difficulty"].astype(float)

    # theta = [difficulty_increment, exploration_sigma, transition_prior_concentration, learning_rate_scale]
    lower = np.array([0.00, 0.00, 0.10, 0.50])
    upper = np.array([0.20, 0.20, 2.00, 2.00])
    ac = cfg["afoa"]
    afoa = AFOA(lower, upper, AFOAConfig(
        population_size=ac["population_size"], max_iterations=ac["max_iterations"],
        w_max=ac["w_max"], w_min=ac["w_min"], c1=ac["c1"], c2=ac["c2"],
        levy_beta=ac["levy_beta"], levy_alpha=ac["levy_alpha"], diversity_threshold=ac["diversity_threshold"],
        finite_difference_delta=ac["finite_difference_delta"], local_step_size=ac["local_step_size"], seed=cfg["seed"],
    ))
    result = afoa.optimize(lambda theta: simulated_proficiency_gain(theta, difficulties, seed=cfg["seed"]))
    serializable = {
        "best_position": result["best_position"].tolist(),
        "best_fitness": result["best_fitness"],
        "history": result["history"],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps({"best_position": serializable["best_position"], "best_fitness": serializable["best_fitness"]}, indent=2))


if __name__ == "__main__":
    main()
