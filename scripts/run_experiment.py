"""
Orchestrator for experiment scripts.

Run a script with its config:

```bash
uv run -m scripts.run_experiment grf_rollout_stats=2026-06-30-smoke
```

By default, no script runs.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TypeAlias

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

ScriptMain: TypeAlias = Callable[[DictConfig], object]

SCRIPTS = {
    "grf_rollout_stats": "scripts.grf_rollout_stats:main",
    "env_mechanism_probes": "scripts.env_mechanism_probes:main",
    "cross_env_compute_sharing": "scripts.cross_env_compute_sharing:main",
    "internal_representation_geometry": "scripts.internal_representation_geometry:main",
    "normalize_soccer_data": "scripts.normalize_soccer_data:main",
    "compare_soccer_stats": "scripts.compare_soccer_stats:main",
    "analyze_fuji_soccer_samples": "scripts.analyze_fuji_soccer_samples:main",
    "download_stp_tracking_sample": "scripts.download_stp_tracking_sample:main",
}


def _load_script(target: str) -> ScriptMain:
    module_name, attribute = target.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, attribute)


@hydra.main(config_path="../configs", config_name="run_experiment.yaml", version_base=None)
def main(cfg: DictConfig):
    selected = [name for name in SCRIPTS if OmegaConf.select(cfg, name, default=None) is not None]

    if not selected:
        logger.info("No script specified; nothing to run")
        return
    if len(selected) > 1:
        logger.error(f"Only one script per run; got {selected}")
        raise SystemExit(1)

    name = selected[0]
    _load_script(SCRIPTS[name])(cfg)


if __name__ == "__main__":
    main()
