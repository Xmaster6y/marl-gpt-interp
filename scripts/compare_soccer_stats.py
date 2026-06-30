"""Compare normalized soccer statistics across datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.soccer_metrics import compare_datasets
from marl_gpt_interp.soccer_schema import read_rows, write_csv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _read_many(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_rows(_resolve(root, path)))
    return rows


def _json_default(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool, list, dict)) else str(value)


def main(cfg: DictConfig) -> list[dict[str, Any]]:
    script_cfg = cfg.compare_soccer_stats
    root = _repo_root()
    event_paths = list(OmegaConf.select(script_cfg, "event_paths", default=[]))
    tracking_paths = list(OmegaConf.select(script_cfg, "tracking_paths", default=[]))
    output_dir = _resolve(root, str(script_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = compare_datasets(
        event_rows=_read_many(root, event_paths),
        tracking_rows=_read_many(root, tracking_paths),
        grid_x=int(OmegaConf.select(script_cfg, "pitch_control.grid_x", default=12)),
        grid_y=int(OmegaConf.select(script_cfg, "pitch_control.grid_y", default=8)),
    )

    with (output_dir / "comparison.json").open("w") as handle:
        json.dump(rows, handle, default=_json_default, indent=2, sort_keys=True)
    write_csv(output_dir / "comparison.csv", rows)
    logger.info(f"Wrote soccer statistics comparison for {len(rows)} datasets to {output_dir}")
    return rows
