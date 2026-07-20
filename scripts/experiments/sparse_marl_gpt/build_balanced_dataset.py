"""Build resumable, balanced offline MARL-GPT dataset views on JZ SCRATCH."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.balanced_dataset import materialize_balanced_view


@hydra.main(config_path="../../../configs/experiments/sparse_marl_gpt/build_balanced_dataset", version_base=None)
def main(cfg: DictConfig) -> dict:
    return materialize_balanced_view(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
