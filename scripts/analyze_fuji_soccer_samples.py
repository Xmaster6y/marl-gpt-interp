"""Analyze the bounded La Liga and RoboCup samples copied from Fuji."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from marl_gpt_interp.soccer_sample_analysis import (
    analyze_laliga_arrow,
    analyze_robocup_pair,
    analyze_stp_tracking_csv,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


@hydra.main(config_path="../configs/analyze_fuji_soccer_samples", version_base=None)
def main(cfg: DictConfig) -> dict:
    script_cfg = cfg
    root = _repo_root()
    output_path = _resolve(root, str(script_cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "laliga": analyze_laliga_arrow(_resolve(root, str(script_cfg.laliga_arrow_path))),
        "robocup2d": [
            analyze_robocup_pair(_resolve(root, str(pair.left)), _resolve(root, str(pair.right)))
            for pair in script_cfg.robocup_pairs
        ],
        "stp_raw": analyze_stp_tracking_csv(
            _resolve(root, str(script_cfg.stp_raw_tracking_path)),
            _resolve(root, str(script_cfg.stp_derived_left_path)),
        ),
    }
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    logger.info(f"Wrote Fuji soccer sample diagnostics to {output_path}")
    return result


if __name__ == "__main__":
    main()
