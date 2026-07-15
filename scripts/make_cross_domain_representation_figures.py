"""Create Matplotlib figures for cross-environment and cross-football latents."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from marl_gpt_interp.marl_gpt_tools import as_path, repo_root


LAYERS = list(range(7))
POOLS = ("mean", "final")
COLORS = ("#2F6FDB", "#C75146", "#2F8F5B")
CKA_YMAX = 1.02


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def filtered_rows(path: Path, sample_unit: str | None) -> list[dict[str, str]]:
    rows = read_csv(path)
    if sample_unit is None:
        return rows
    return [row for row in rows if row.get("sample_unit") == sample_unit]


def layer_series(
    rows: list[dict[str, str]],
    *,
    group_column: str,
    group: str,
    pool: str,
    value_column: str,
) -> list[float]:
    values = []
    for layer in LAYERS:
        feature = f"layer_{layer:02d}:{pool}"
        matches = [
            row
            for row in rows
            if row.get(group_column) == group and row.get("feature") == feature
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {group=} {feature=} row, found {len(matches)}")
        values.append(float(matches[0][value_column]))
    return values


def plot_lines(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    groups: list[str],
    labels: dict[str, str],
    group_column: str,
    pool: str,
    value_column: str,
) -> None:
    for group, color in zip(groups, COLORS, strict=True):
        ax.plot(
            LAYERS,
            layer_series(
                rows,
                group_column=group_column,
                group=group,
                pool=pool,
                value_column=value_column,
            ),
            color=color,
            label=labels[group],
            linewidth=1.8,
            marker="o",
            markersize=3.2,
        )
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Transformer layer")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    written = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path.name)
    plt.close(fig)
    return written


def cosine_figure(
    cross_env_dir: Path,
    football_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    cross_env = filtered_rows(
        cross_env_dir / "activation_pairwise_cosine_similarity.csv",
        None,
    )
    football = filtered_rows(
        football_dir / "activation_pairwise_cosine_similarity.csv",
        "frame_mean",
    )
    env_pairs = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
    env_labels = {
        "smac_vs_pogema": "SMAC–POGEMA",
        "smac_vs_grf": "SMAC–GRF",
        "pogema_vs_grf": "POGEMA–GRF",
    }
    football_pairs = ["laliga_vs_robocup", "laliga_vs_grf", "robocup_vs_grf"]
    football_labels = {
        "laliga_vs_robocup": "La Liga–RoboCup",
        "laliga_vs_grf": "La Liga–GRF",
        "robocup_vs_grf": "RoboCup–GRF",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True)
    for column, pool in enumerate(POOLS):
        plot_lines(
            axes[0, column],
            cross_env,
            groups=env_pairs,
            labels=env_labels,
            group_column="env_pair",
            pool=pool,
            value_column="mean_cosine_similarity",
        )
        plot_lines(
            axes[1, column],
            football,
            groups=football_pairs,
            labels=football_labels,
            group_column="env_pair",
            pool=pool,
            value_column="mean_cosine_similarity",
        )
        axes[0, column].set_title(f"Cross-environment · {pool}")
        axes[1, column].set_title(f"Cross-football · {pool}")
    for ax in axes.flat:
        ax.set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("Mean pairwise cosine")
    axes[1, 0].set_ylabel("Mean pairwise cosine")
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[1, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("Layerwise cross-domain activation cosine", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, output_dir, "layerwise_pairwise_cosine", formats)


def cka_figure(
    source_dir: Path,
    output_dir: Path,
    formats: list[str],
    *,
    stem: str,
    title: str,
    pairs: list[str],
    pair_labels: dict[str, str],
    sources: list[str],
    source_labels: dict[str, str],
    sample_unit: str | None,
    ymax: float,
) -> list[str]:
    cross_rows = filtered_rows(source_dir / "activation_subspace_similarity.csv", sample_unit)
    self_rows = filtered_rows(source_dir / "self_subspace_similarity.csv", sample_unit)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True, sharey=True)
    for column, pool in enumerate(POOLS):
        plot_lines(
            axes[0, column],
            cross_rows,
            groups=pairs,
            labels=pair_labels,
            group_column="env_pair",
            pool=pool,
            value_column="linear_cka",
        )
        plot_lines(
            axes[1, column],
            self_rows,
            groups=sources,
            labels=source_labels,
            group_column="env",
            pool=pool,
            value_column="linear_cka",
        )
        axes[0, column].set_title(f"Cross-source · {pool}")
        axes[1, column].set_title(f"Within-source self-CKA · {pool}")
    for ax in axes.flat:
        ax.set_ylim(0, ymax)
    axes[0, 0].set_ylabel("Centered linear CKA")
    axes[1, 0].set_ylabel("Centered linear CKA")
    legend_location = "lower left" if ymax > 0.5 else "upper left"
    axes[0, 0].legend(frameon=False, loc=legend_location)
    axes[1, 0].legend(frameon=False, loc=legend_location)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, formats)


@hydra.main(config_path="../configs/make_cross_domain_representation_figures", version_base=None)
def main(cfg: DictConfig) -> dict[str, Any]:
    configure_style()
    root = repo_root()
    output_dir = as_path(root, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_env_dir = as_path(root, str(cfg.cross_env_dir))
    football_dir = as_path(root, str(cfg.cross_football_dir))
    formats = [str(value) for value in cfg.formats]

    files = []
    files.extend(cosine_figure(cross_env_dir, football_dir, output_dir, formats))
    files.extend(
        cka_figure(
            cross_env_dir,
            output_dir,
            formats,
            stem="cross_environment_cka",
            title="SMAC, POGEMA, and GRF representation geometry",
            pairs=["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"],
            pair_labels={
                "smac_vs_pogema": "SMAC–POGEMA",
                "smac_vs_grf": "SMAC–GRF",
                "pogema_vs_grf": "POGEMA–GRF",
            },
            sources=["smac", "pogema", "grf"],
            source_labels={"smac": "SMAC", "pogema": "POGEMA", "grf": "GRF"},
            sample_unit=None,
            ymax=CKA_YMAX,
        )
    )
    files.extend(
        cka_figure(
            football_dir,
            output_dir,
            formats,
            stem="cross_football_cka",
            title="La Liga, RoboCup, and GRF representation geometry",
            pairs=["laliga_vs_robocup", "laliga_vs_grf", "robocup_vs_grf"],
            pair_labels={
                "laliga_vs_robocup": "La Liga–RoboCup",
                "laliga_vs_grf": "La Liga–GRF",
                "robocup_vs_grf": "RoboCup–GRF",
            },
            sources=["laliga", "robocup", "grf"],
            source_labels={"laliga": "La Liga", "robocup": "RoboCup", "grf": "GRF"},
            sample_unit="frame_mean",
            ymax=CKA_YMAX,
        )
    )
    manifest = {
        "cross_env_dir": str(cross_env_dir),
        "cross_football_dir": str(football_dir),
        "pooling": "mean excludes final position; final-token panels retained",
        "football_sample_unit": "frame_mean",
        "files": files,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(files)} figures to {output_dir}")
    return manifest


if __name__ == "__main__":
    main()
