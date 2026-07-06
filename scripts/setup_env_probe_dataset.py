"""Download the small offline dataset subset used by env_mechanism_probes."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/nortem/marl-gpt-datasets/resolve/main"

FILES = [
    (
        "dataset-SMACv2/zerg_5_vs_5/chunk_0_part_0.pt",
        "dataset/zerg_5_vs_5/chunk_0_part_0.pt",
    ),
    (
        "dataset-LMAPF/random/part_0_0.arrow",
        "dataset/dataset_pogema_ll/random/part_0_0.arrow",
    ),
    (
        "dataset-GRF/academy_corner/chunk_1_part_0.pt",
        "dataset/dataset_grf/trajectories/academy_corner/chunk_1_part_0.pt",
    ),
]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"download {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=60) as response:
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    for source, target in FILES:
        download_file(f"{BASE_URL}/{source}", args.root / target)


if __name__ == "__main__":
    main()
