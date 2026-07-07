"""Create paper-ready activation and parameter analysis figures.

The figures are intentionally limited to quantitative results. They exclude
method schemas and slide-style summary graphics.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CROSS_ENV_DIR = ROOT / "results/experiments/2026-07-06-cross-env-compute-sharing"
GEOMETRY_DIR = ROOT / "results/experiments/2026-07-06-internal-representation-geometry"
OUT_DIR = ROOT / "results/figures/paper-ready-act-param"

PAIR_ORDER = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
PAIR_LABELS = {
    "smac_vs_pogema": "SMAC-POGEMA",
    "smac_vs_grf": "SMAC-GRF",
    "pogema_vs_grf": "POGEMA-GRF",
}
ENV_ORDER = ["smac", "pogema", "grf"]
ENV_LABELS = {"smac": "SMAC", "pogema": "POGEMA", "grf": "GRF"}
COLORS = {
    "smac": "#2F6FDB",
    "pogema": "#C75146",
    "grf": "#2F8F5B",
    "smac_vs_pogema": "#7B5BB7",
    "smac_vs_grf": "#1F7A8C",
    "pogema_vs_grf": "#D1862F",
}
PARAM_GROUPS = [
    "token_embeddings",
    "layer_00",
    "layer_01",
    "layer_02",
    "layer_03",
    "layer_04",
    "layer_05",
    "layer_06",
    "critic_head",
    "actor_head",
]
PARAM_LABELS = {
    "token_embeddings": "Tok.",
    "layer_00": "L0",
    "layer_01": "L1",
    "layer_02": "L2",
    "layer_03": "L3",
    "layer_04": "L4",
    "layer_05": "L5",
    "layer_06": "L6",
    "critic_head": "Critic",
    "actor_head": "Actor",
}
LAYERS = list(range(7))
ENV_TO_ID = {"smac": 1, "pogema": 2, "grf": 3}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def env_pair(left_env: str, right_env: str) -> str:
    left_id = ENV_TO_ID[left_env]
    right_id = ENV_TO_ID[right_env]
    if left_id < right_id:
        return f"{left_env}_vs_{right_env}"
    return f"{right_env}_vs_{left_env}"


def matrix_plot(
    matrix: list[list[float]],
    *,
    title: str,
    colorbar_label: str,
    stem: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(ENV_ORDER)), [ENV_LABELS[env] for env in ENV_ORDER])
    ax.set_yticks(range(len(ENV_ORDER)), [ENV_LABELS[env] for env in ENV_ORDER])
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > (vmin + vmax) / 2 else "#111827"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    save_figure(fig, stem)


def line_for_pair(ax: plt.Axes, x_values: list[int], values: list[float], pair: str, *, label: bool = True) -> None:
    ax.plot(
        x_values,
        values,
        marker="o",
        markersize=3.2,
        linewidth=1.8,
        color=COLORS[pair],
        label=PAIR_LABELS[pair] if label else None,
    )


def plot_parameter_gradient_cosines() -> None:
    rows = [
        row
        for row in read_csv(CROSS_ENV_DIR / "parameter_gradient_overlap.csv")
        if row["direction_type"] == "gradient_cosine" and row["parameter_group"] in PARAM_GROUPS
    ]
    values = {(row["env_pair"], row["parameter_group"]): float(row["cosine"]) for row in rows}
    x = list(range(len(PARAM_GROUPS)))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for pair in PAIR_ORDER:
        line_for_pair(ax, x, [values[(pair, group)] for group in PARAM_GROUPS], pair)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, [PARAM_LABELS[group] for group in PARAM_GROUPS])
    ax.set_ylim(-0.12, 1.03)
    ax.set_ylabel("Gradient cosine")
    ax.set_title("Parameter-gradient alignment by model component")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save_figure(fig, "parameter_gradient_cosines")


def plot_parameter_gradient_norms() -> None:
    rows = [
        row
        for row in read_csv(CROSS_ENV_DIR / "parameter_gradients.csv")
        if row["parameter_group"] in PARAM_GROUPS
    ]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["parameter_group"])].append(float(row["gradient_l2"]))

    x = list(range(len(PARAM_GROUPS)))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, env in enumerate(ENV_ORDER):
        offsets = [v + (i - 1) * width for v in x]
        heights = [mean(grouped[(env, group)]) for group in PARAM_GROUPS]
        ax.bar(offsets, heights, width=width, color=COLORS[env], label=ENV_LABELS[env])
    ax.set_xticks(x, [PARAM_LABELS[group] for group in PARAM_GROUPS])
    ax.set_yscale("log")
    ax.set_ylabel("Gradient L2 norm (log scale)")
    ax.set_title("Environment-specific gradient magnitude by component")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    save_figure(fig, "parameter_gradient_norms")


def activation_value(rows: list[dict[str, str]], pair: str, layer: int, pool: str, column: str) -> float:
    feature = f"layer_{layer:02d}:{pool}"
    for row in rows:
        if row["env_pair"] == pair and row["feature"] == feature:
            return float(row[column])
    raise KeyError((pair, feature, column))


def plot_activation_cka_layers() -> None:
    rows = read_csv(GEOMETRY_DIR / "activation_subspace_similarity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    for ax, pool in zip(axes, ["mean", "final"], strict=True):
        for pair in PAIR_ORDER:
            values = [activation_value(rows, pair, layer, pool, "linear_cka") for layer in LAYERS]
            line_for_pair(ax, LAYERS, values, pair, label=(pool == "mean"))
        ax.set_title(f"{pool}-pooled states")
        ax.set_xlabel("Transformer layer")
        ax.set_xticks(LAYERS)
        ax.set_ylim(0, 0.10)
    axes[0].set_ylabel("Centered linear CKA")
    axes[0].legend(ncol=1, frameon=False, loc="upper left")
    fig.suptitle("Activation subspace similarity is low across environments", y=1.03, fontsize=9)
    save_figure(fig, "activation_cka_layers")


def plot_activation_cka_matrix_if_available() -> None:
    self_path = GEOMETRY_DIR / "self_subspace_similarity.csv"
    cross_path = GEOMETRY_DIR / "activation_subspace_similarity.csv"
    if not self_path.exists():
        return
    self_rows = read_csv(self_path)
    cross_rows = read_csv(cross_path)
    self_by_env = defaultdict(list)
    for row in self_rows:
        if row["feature"].startswith("layer_") or row["feature"] in {"actor_layer:final", "critic_layer:final"}:
            self_by_env[row["env"]].append(float(row["linear_cka"]))
    cross_by_pair = defaultdict(list)
    for row in cross_rows:
        if row["feature"].startswith("layer_") or row["feature"] in {"actor_layer:final", "critic_layer:final"}:
            cross_by_pair[row["env_pair"]].append(float(row["linear_cka"]))
    matrix = []
    for left_env in ENV_ORDER:
        row_values = []
        for right_env in ENV_ORDER:
            if left_env == right_env:
                row_values.append(mean(self_by_env[left_env]))
            else:
                row_values.append(mean(cross_by_pair[env_pair(left_env, right_env)]))
        matrix.append(row_values)
    matrix_plot(
        matrix,
        title="Self-CKA diagonal vs cross-environment CKA",
        colorbar_label="Centered linear CKA",
        stem="activation_cka_self_cross_matrix",
        vmin=0,
        vmax=1,
    )


def plot_activation_separation_layers() -> None:
    rows = read_csv(GEOMETRY_DIR / "representation_separation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    for ax, pool in zip(axes, ["mean", "final"], strict=True):
        for pair in PAIR_ORDER:
            values = [activation_value(rows, pair, layer, pool, "normalized_centroid_l2") for layer in LAYERS]
            line_for_pair(ax, LAYERS, values, pair, label=(pool == "mean"))
        ax.set_title(f"{pool}-pooled states")
        ax.set_xlabel("Transformer layer")
        ax.set_xticks(LAYERS)
        ax.set_ylim(0, 26)
    axes[0].set_ylabel("Centroid distance / within-env spread")
    axes[0].legend(ncol=1, frameon=False, loc="upper left")
    fig.suptitle("Layerwise representations remain environment-separated", y=1.03, fontsize=9)
    save_figure(fig, "activation_normalized_separation_layers")


def plot_gradient_similarity_matrix_if_available() -> None:
    self_path = CROSS_ENV_DIR / "parameter_gradient_self_similarity.csv"
    cross_path = CROSS_ENV_DIR / "parameter_gradient_overlap.csv"
    if not self_path.exists():
        return
    self_rows = [
        row
        for row in read_csv(self_path)
        if row["parameter_group"] in PARAM_GROUPS and row["direction_type"] == "same_env_gradient_cosine"
    ]
    cross_rows = [
        row
        for row in read_csv(cross_path)
        if row["parameter_group"] in PARAM_GROUPS and row["direction_type"] == "gradient_cosine"
    ]
    self_by_env = defaultdict(list)
    for row in self_rows:
        self_by_env[row["env"]].append(float(row["mean_cosine"]))
    cross_by_pair = defaultdict(list)
    for row in cross_rows:
        cross_by_pair[row["env_pair"]].append(float(row["cosine"]))
    matrix = []
    for left_env in ENV_ORDER:
        row_values = []
        for right_env in ENV_ORDER:
            if left_env == right_env:
                row_values.append(mean(self_by_env[left_env]))
            else:
                row_values.append(mean(cross_by_pair[env_pair(left_env, right_env)]))
        matrix.append(row_values)
    matrix_plot(
        matrix,
        title="Same-env gradient reliability vs cross-env gradients",
        colorbar_label="Gradient cosine",
        stem="parameter_gradient_self_cross_matrix",
        vmin=-0.1,
        vmax=1,
        cmap="coolwarm",
    )


def plot_activation_containment_rank16() -> None:
    rows = [
        row
        for row in read_csv(GEOMETRY_DIR / "asymmetric_representation_analysis.csv")
        if int(row["requested_rank"]) == 16
        and row["feature"] in {"layer_06:final", "actor_layer:final", "critic_layer:final"}
    ]
    direction_order = [
        "smac_to_pogema",
        "pogema_to_smac",
        "smac_to_grf",
        "grf_to_smac",
        "pogema_to_grf",
        "grf_to_pogema",
    ]
    labels = [
        "SMAC→POG.",
        "POG.→SMAC",
        "SMAC→GRF",
        "GRF→SMAC",
        "POG.→GRF",
        "GRF→POG.",
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["direction"]].append(float(row["target_variance_explained"]))

    values = [mean(grouped[direction]) for direction in direction_order]
    colors = [
        COLORS["smac_vs_pogema"],
        COLORS["smac_vs_pogema"],
        COLORS["smac_vs_grf"],
        COLORS["smac_vs_grf"],
        COLORS["pogema_vs_grf"],
        COLORS["pogema_vs_grf"],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Target variance explained")
    ax.set_title("Rank-16 asymmetric containment in final/branch states")
    save_figure(fig, "activation_asymmetric_containment_r16")


def plot_activation_compactness() -> None:
    rows = read_csv(GEOMETRY_DIR / "internal_representation_proximity.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if float(row["mean_pairwise_l2"]) > 1e-6:
            grouped[row["env"]].append(float(row["mean_pairwise_cosine_distance"]))

    values = [sorted(grouped[env])[len(grouped[env]) // 2] for env in ENV_ORDER]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([ENV_LABELS[env] for env in ENV_ORDER], values, color=[COLORS[env] for env in ENV_ORDER])
    ax.set_ylabel("Median pairwise cosine distance")
    ax.set_title("Within-environment activation compactness")
    save_figure(fig, "activation_internal_compactness")


def main() -> None:
    configure_style()
    plot_parameter_gradient_cosines()
    plot_parameter_gradient_norms()
    plot_gradient_similarity_matrix_if_available()
    plot_activation_cka_layers()
    plot_activation_cka_matrix_if_available()
    plot_activation_separation_layers()
    plot_activation_containment_rank16()
    plot_activation_compactness()
    print(f"Wrote paper-ready figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
