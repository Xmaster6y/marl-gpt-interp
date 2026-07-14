"""Normalize provider-specific soccer event/tracking data into JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.soccer_schema import normalize_rows, read_rows, write_jsonl


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True) or {}
    return dict(value)


def _normalize_source(root: Path, source: DictConfig) -> dict[str, Any]:
    input_path = _resolve(root, str(source.input_path))
    output_path = _resolve(root, str(source.output_path))
    rows = read_rows(input_path)
    normalized = normalize_rows(
        rows,
        dataset=str(source.dataset),
        kind=str(source.kind),
        column_map=_to_plain_dict(source.column_map),
        constants=_to_plain_dict(OmegaConf.select(source, "constants", default={})),
    )
    write_jsonl(output_path, normalized)
    logger.info(f"Wrote {len(normalized)} normalized {source.kind} rows for {source.dataset} to {output_path}")
    return {
        "dataset": str(source.dataset),
        "kind": str(source.kind),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(normalized),
    }


@hydra.main(config_path="../configs/normalize_soccer_data", version_base=None)
def main(cfg: DictConfig) -> list[dict[str, Any]]:
    script_cfg = cfg
    root = _repo_root()
    return [_normalize_source(root, source) for source in script_cfg.sources]


if __name__ == "__main__":
    main()
