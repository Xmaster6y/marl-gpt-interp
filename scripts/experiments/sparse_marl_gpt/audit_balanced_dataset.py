"""Audit materialized balanced-corpus views before activation collection."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from marl_gpt_interp.balanced_dataset import audit_balanced_view


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/audit_balanced_dataset", version_base=None)
def main(cfg: DictConfig) -> dict:
    return audit_balanced_view(Path(str(cfg.manifest_path)), Path(str(cfg.output_path)))


if __name__ == "__main__":
    main()
