"""Create lightweight SVG figures for the cross-environment representation results."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
CROSS_ENV_DIR = ROOT / "results/experiments/2026-07-06-cross-env-compute-sharing"
GEOMETRY_DIR = ROOT / "results/experiments/2026-07-06-internal-representation-geometry"
OUT_DIR = ROOT / "results/figures/2026-07-06-environment-separated-partial-sharing"

COLORS = {
    "smac": "#2f6fbb",
    "pogema": "#c75146",
    "grf": "#2f8f5b",
    "smac_vs_pogema": "#7b5bb7",
    "smac_vs_grf": "#1f7a8c",
    "pogema_vs_grf": "#d1862f",
    "smac_to_pogema": "#7b5bb7",
    "pogema_to_smac": "#b68bd7",
    "smac_to_grf": "#1f7a8c",
    "grf_to_smac": "#6fb6c4",
    "pogema_to_grf": "#d1862f",
    "grf_to_pogema": "#e7b86b",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def svg_page(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2933; }}
.title {{ font-size: 24px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #52606d; }}
.axis {{ stroke: #9aa5b1; stroke-width: 1; }}
.grid {{ stroke: #d9e2ec; stroke-width: 1; }}
.label {{ font-size: 12px; fill: #334e68; }}
.small {{ font-size: 11px; fill: #52606d; }}
</style>
{body}
</svg>
"""


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_page(width, height, body))


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "axis") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def title_block(title: str, subtitle: str) -> str:
    return text(48, 42, title, "title") + text(48, 66, subtitle, "subtitle")


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def parameter_group_order(name: str) -> int:
    order = {
        "token_embeddings": 0,
        "other": 1,
        "layer_00": 2,
        "layer_01": 3,
        "layer_02": 4,
        "layer_03": 5,
        "layer_04": 6,
        "layer_05": 7,
        "layer_06": 8,
        "critic_head": 9,
        "actor_head": 10,
    }
    return order.get(name, 99)


def feature_layer_index(feature: str) -> int | None:
    if feature.startswith("layer_"):
        return int(feature.split(":", maxsplit=1)[0].split("_")[1])
    return None


def feature_kind(feature: str) -> str:
    return feature.split(":", maxsplit=1)[1] if ":" in feature else feature


def bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    y_label: str,
    y_max: float | None = None,
) -> None:
    width, height = 980, 560
    left, right, top, bottom = 86, 40, 104, 96
    chart_w, chart_h = width - left - right, height - top - bottom
    max_value = y_max if y_max is not None else max(values) * 1.1
    body = [title_block(title, subtitle)]
    for i in range(5):
        v = max_value * i / 4
        y = top + chart_h - chart_h * i / 4
        body.append(line(left, y, left + chart_w, y, "grid"))
        body.append(text(left - 10, y + 4, f"{v:.2f}", "small", "end"))
    body.append(line(left, top, left, top + chart_h))
    body.append(line(left, top + chart_h, left + chart_w, top + chart_h))
    body.append(text(22, top + chart_h / 2, y_label, "small", "middle"))
    bar_w = chart_w / max(len(values), 1) * 0.68
    for i, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        x = left + (i + 0.16) * chart_w / len(values)
        y = scale(value, 0, max_value, top + chart_h, top)
        h = top + chart_h - y
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(text(x + bar_w / 2, y - 8, f"{value:.2f}", "small", "middle"))
        body.append(text(x + bar_w / 2, top + chart_h + 22, label, "small", "middle"))
    write_svg(path, width, height, "\n".join(body))


def grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    groups: list[str],
    series: dict[str, list[float]],
    y_label: str,
    y_min: float,
    y_max: float,
) -> None:
    width, height = 1160, 600
    left, right, top, bottom = 92, 42, 112, 124
    chart_w, chart_h = width - left - right, height - top - bottom
    body = [title_block(title, subtitle)]
    for i in range(6):
        v = y_min + (y_max - y_min) * i / 5
        y = scale(v, y_min, y_max, top + chart_h, top)
        body.append(line(left, y, left + chart_w, y, "grid"))
        body.append(text(left - 10, y + 4, f"{v:.1f}", "small", "end"))
    body.append(line(left, top, left, top + chart_h))
    body.append(line(left, top + chart_h, left + chart_w, top + chart_h))
    body.append(text(24, top + chart_h / 2, y_label, "small", "middle"))
    names = list(series)
    slot_w = chart_w / len(groups)
    bar_w = slot_w / (len(names) + 1)
    for gi, group in enumerate(groups):
        base_x = left + gi * slot_w
        body.append(text(base_x + slot_w / 2, top + chart_h + 24, group.replace("_", " "), "small", "middle"))
        for si, name in enumerate(names):
            value = series[name][gi]
            x = base_x + (si + 0.5) * bar_w
            y = scale(value, y_min, y_max, top + chart_h, top)
            h = top + chart_h - y
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{h:.1f}" fill="{COLORS[name]}"/>'
            )
    legend_x = left + 8
    for i, name in enumerate(names):
        x = legend_x + i * 190
        body.append(f'<rect x="{x:.1f}" y="82" width="14" height="14" fill="{COLORS[name]}"/>')
        body.append(text(x + 22, 94, name.replace("_", " "), "small"))
    write_svg(path, width, height, "\n".join(body))


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_label: str,
    y_min: float,
    y_max: float,
) -> None:
    width, height = 1060, 600
    left, right, top, bottom = 86, 42, 108, 88
    chart_w, chart_h = width - left - right, height - top - bottom
    body = [title_block(title, subtitle)]
    for i in range(6):
        v = y_min + (y_max - y_min) * i / 5
        y = scale(v, y_min, y_max, top + chart_h, top)
        body.append(line(left, y, left + chart_w, y, "grid"))
        body.append(text(left - 10, y + 4, f"{v:.2f}", "small", "end"))
    body.append(line(left, top, left, top + chart_h))
    body.append(line(left, top + chart_h, left + chart_w, top + chart_h))
    body.append(text(24, top + chart_h / 2, y_label, "small", "middle"))
    for x_value in x_values:
        x = scale(x_value, min(x_values), max(x_values), left, left + chart_w)
        body.append(line(x, top + chart_h, x, top + chart_h + 5))
        body.append(text(x, top + chart_h + 24, str(x_value), "small", "middle"))
    for name, values in series.items():
        points = []
        for x_value, value in zip(x_values, values, strict=True):
            x = scale(x_value, min(x_values), max(x_values), left, left + chart_w)
            y = scale(value, y_min, y_max, top + chart_h, top)
            points.append((x, y))
        path_data = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        body.append(f'<path d="{path_data}" fill="none" stroke="{COLORS[name]}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[name]}"/>')
    for i, name in enumerate(series):
        x = left + i * 190
        body.append(f'<rect x="{x:.1f}" y="82" width="14" height="14" fill="{COLORS[name]}"/>')
        body.append(text(x + 22, 94, name.replace("_", " "), "small"))
    write_svg(path, width, height, "\n".join(body))


def heatmap(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[str],
    cols: list[str],
    values: dict[tuple[str, str], float],
    value_min: float,
    value_max: float,
) -> None:
    width, height = 1160, 640
    left, right, top, bottom = 220, 52, 120, 72
    cell_w = (width - left - right) / len(cols)
    cell_h = (height - top - bottom) / len(rows)
    body = [title_block(title, subtitle)]
    for ci, col in enumerate(cols):
        body.append(text(left + ci * cell_w + cell_w / 2, top - 14, col.replace("_", " "), "small", "middle"))
    for ri, row in enumerate(rows):
        body.append(text(left - 12, top + ri * cell_h + cell_h * 0.62, row.replace("_", " "), "small", "end"))
        for ci, col in enumerate(cols):
            value = values[(row, col)]
            t = max(0.0, min(1.0, (value - value_min) / (value_max - value_min)))
            red = int(246 - t * 205)
            green = int(248 - t * 112)
            blue = int(250 - t * 91)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            x = left + ci * cell_w
            y = top + ri * cell_h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" fill="{fill}"/>'
            )
            body.append(text(x + cell_w / 2, y + cell_h * 0.62, f"{value:.2f}", "small", "middle"))
    write_svg(path, width, height, "\n".join(body))


def methodology_schema() -> None:
    width, height = 1200, 680
    body = [
        title_block(
            "Methodology schema",
            "Natural-inference probes for shared versus environment-specific computation",
        )
    ]
    boxes = [
        ("Datasets", "SMAC\\nPOGEMA\\nGRF", 70, 140, 180, 130),
        ("MARL-GPT", "Correct env token\\nnormal obs/mask\\nnatural inference", 320, 140, 220, 130),
        ("Capture", "embed\\nlayers 0-6\\nactor/critic branches", 620, 140, 220, 130),
        ("Representation analyses", "CKA\\ninternal proximity\\nasymmetric containment", 170, 380, 260, 150),
        ("Effective computation", "per-env gradients\\ngradient cosine\\nparameter groups", 500, 380, 240, 150),
        ("Claims", "separation\\npartial sharing\\nnext: concepts/patching", 820, 380, 260, 150),
    ]
    for title, detail, x, y, w, h in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#f8fafc" stroke="#9fb3c8"/>')
        body.append(text(x + 16, y + 28, title, "label"))
        for i, part in enumerate(detail.split("\\n")):
            body.append(text(x + 16, y + 58 + i * 22, part, "small"))
    arrows = [
        (250, 205, 320, 205),
        (540, 205, 620, 205),
        (730, 270, 300, 380),
        (730, 270, 620, 380),
        (740, 455, 820, 455),
        (430, 455, 500, 455),
    ]
    for x1, y1, x2, y2 in arrows:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#52606d" stroke-width="2"/>')
        angle = math.atan2(y2 - y1, x2 - x1)
        a1, a2 = angle + math.pi * 0.82, angle - math.pi * 0.82
        p1 = (x2 + 10 * math.cos(a1), y2 + 10 * math.sin(a1))
        p2 = (x2 + 10 * math.cos(a2), y2 + 10 * math.sin(a2))
        body.append(
            f'<path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} '
            f'L{p2[0]:.1f},{p2[1]:.1f} Z" fill="#52606d"/>'
        )
    body.append(
        text(
            72,
            600,
            "Rule: geometry says whether spaces are comparable; gradients say whether parameters are used similarly.",
            "subtitle",
        )
    )
    write_svg(OUT_DIR / "methodology-schema.svg", width, height, "\n".join(body))


def arrow(body: list[str], x1: float, y1: float, x2: float, y2: float, color: str = "#52606d", width: int = 2) -> None:
    body.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )
    angle = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = angle + math.pi * 0.82, angle - math.pi * 0.82
    p1 = (x2 + 12 * math.cos(a1), y2 + 12 * math.sin(a1))
    p2 = (x2 + 12 * math.cos(a2), y2 + 12 * math.sin(a2))
    body.append(
        f'<path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" '
        f'fill="{color}"/>'
    )


def rounded_box(
    body: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    fill: str,
    stroke: str,
) -> None:
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    body.append(text(x + 18, y + 30, title, "label"))
    for i, line_value in enumerate(lines):
        body.append(text(x + 18, y + 60 + i * 23, line_value, "small"))


def env_identity_corruption_schema() -> None:
    width, height = 1280, 760
    body = [
        title_block(
            "Environment identity corruption probe",
            "Test whether hidden states follow the prompt token or recover the true environment from observations.",
        )
    ]
    envs = [("SMAC", COLORS["smac"], 120), ("POGEMA", COLORS["pogema"], 260), ("GRF", COLORS["grf"], 400)]
    for env, color, y in envs:
        body.append(f'<circle cx="105" cy="{y}" r="24" fill="{color}"/>')
        body.append(text(105, y + 5, env[0], "label", "middle"))
        rounded_box(
            body,
            150,
            y - 42,
            220,
            84,
            f"True {env} sample",
            ["observation stream", "action mask", "positions"],
            "#f8fafc",
            color,
        )
        arrow(body, 370, y, 445, y, color, 3)
    rounded_box(
        body,
        445,
        168,
        240,
        190,
        "Corrupt final token",
        ["keep true obs/mask", "replace env token", "run same model"],
        "#fff7ed",
        "#d1862f",
    )
    for _env, color, y in envs:
        arrow(body, 685, 263, 780, y, color, 3)
    rounded_box(
        body,
        780,
        88,
        250,
        124,
        "Correct token",
        ["baseline activations", "baseline action logits"],
        "#eff6ff",
        "#2f6fbb",
    )
    rounded_box(
        body,
        780,
        258,
        250,
        124,
        "Wrong token",
        ["counterfactual activations", "logit/value shifts"],
        "#fff1f2",
        "#c75146",
    )
    rounded_box(
        body,
        780,
        428,
        250,
        124,
        "Token sweep",
        ["force SMAC / POGEMA / GRF", "compare prompt sensitivity"],
        "#f0fdf4",
        "#2f8f5b",
    )
    arrow(body, 1030, 150, 1110, 250, "#52606d", 2)
    arrow(body, 1030, 320, 1110, 320, "#52606d", 2)
    arrow(body, 1030, 490, 1110, 390, "#52606d", 2)
    rounded_box(
        body,
        1110,
        235,
        130,
        185,
        "Readouts",
        ["true-env probe", "prompt probe", "behavior delta"],
        "#f8fafc",
        "#9fb3c8",
    )
    body.append(
        text(
            80,
            650,
            "Interpretation: true-env decoding under wrong-token prompts means observations carry env identity.",
            "subtitle",
        )
    )
    body.append(
        text(
            80,
            676,
            "If actions/logits shift toward the prompt, the explicit env token has functional control.",
            "subtitle",
        )
    )
    write_svg(OUT_DIR / "env-identity-corruption-schema.svg", width, height, "\n".join(body))


def representation_analysis_schema() -> None:
    width, height = 1380, 820
    body = [
        title_block(
            "Representation analysis: what each statistic means",
            "Cross-env CKA is not averaged inside each layer curve; self-CKA is the split-half reliability baseline.",
        )
    ]
    rounded_box(
        body,
        70,
        120,
        250,
        145,
        "Activation matrices",
        ["A_env, layer, pool", "rows = examples", "cols = hidden dims"],
        "#f8fafc",
        "#9fb3c8",
    )
    rounded_box(
        body,
        390,
        70,
        300,
        150,
        "Cross-env CKA",
        ["CKA(A_smac, A_grf)", "computed per layer + pool", "curve = exact row value"],
        "#eff6ff",
        "#2f6fbb",
    )
    rounded_box(
        body,
        390,
        260,
        300,
        150,
        "Self-CKA",
        ["split env examples in half", "CKA(A_env half 1, half 2)", "baseline for reliability"],
        "#f0fdf4",
        "#2f8f5b",
    )
    rounded_box(
        body,
        390,
        450,
        300,
        150,
        "Internal proximity",
        ["pairwise distance", "distance to centroid", "PCA effective rank"],
        "#fff7ed",
        "#d1862f",
    )
    rounded_box(
        body,
        760,
        160,
        310,
        155,
        "Asymmetric containment",
        ["fit PCA basis on source", "project target activations", "rank-k variance explained"],
        "#f5f3ff",
        "#7b5bb7",
    )
    rounded_box(
        body,
        760,
        395,
        310,
        155,
        "Normalized separation",
        ["centroid distance", "/ pooled within-env spread", "plus silhouette + NN identity"],
        "#ecfeff",
        "#1f7a8c",
    )
    rounded_box(
        body,
        1130,
        250,
        190,
        190,
        "Slide summaries",
        ["layer curves: exact", "bar charts: medians", "text means: across rows"],
        "#f8fafc",
        "#52606d",
    )
    for y in [192, 335, 525]:
        arrow(body, 320, 192, 390, y, "#52606d", 2)
    arrow(body, 690, 145, 760, 238, "#2f6fbb", 3)
    arrow(body, 690, 335, 760, 472, "#2f8f5b", 3)
    arrow(body, 1070, 238, 1130, 315, "#7b5bb7", 3)
    arrow(body, 1070, 472, 1130, 375, "#1f7a8c", 3)
    body.append(
        text(
            92,
            690,
            "Current run: cross-env CKA exists; self-CKA is now implemented for the next rerun.",
            "subtitle",
        )
    )
    body.append(
        text(
            92,
            720,
            "Presentation wording: 'CKA curves show per-layer centered linear CKA; medians summarize rows.'",
            "subtitle",
        )
    )
    write_svg(OUT_DIR / "representation-analysis-schema.svg", width, height, "\n".join(body))


def make_gradient_heatmap() -> None:
    rows = read_csv(CROSS_ENV_DIR / "parameter_gradient_overlap.csv")
    cos_rows = [row for row in rows if row["direction_type"] == "gradient_cosine"]
    groups = sorted({row["parameter_group"] for row in cos_rows}, key=parameter_group_order)
    pairs = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
    values = {(row["parameter_group"], row["env_pair"]): float(row["cosine"]) for row in cos_rows}
    heatmap(
        OUT_DIR / "gradient-cosine-heatmap.svg",
        "Effective-computation overlap",
        "Gradient cosine by parameter group. POGEMA-GRF aligns; SMAC is near orthogonal.",
        groups,
        pairs,
        values,
        -0.1,
        1.0,
    )


def make_cka_curve() -> None:
    rows = read_csv(GEOMETRY_DIR / "activation_subspace_similarity.csv")
    pairs = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
    for kind in ["mean", "final"]:
        x_values = list(range(7))
        series: dict[str, list[float]] = {}
        for pair in pairs:
            values = []
            for layer in x_values:
                matches = [
                    float(row["linear_cka"])
                    for row in rows
                    if row["env_pair"] == pair and row["feature"] == f"layer_{layer:02d}:{kind}"
                ]
                values.append(matches[0])
            series[pair] = values
        line_chart(
            OUT_DIR / f"cka-layer-{kind}.svg",
            f"Activation CKA across transformer layers ({kind})",
            f"Each point is one centered linear CKA row for layer_k:{kind}; no layerwise mean/max aggregation.",
            x_values,
            series,
            "linear CKA",
            0.0,
            0.1,
        )


def make_internal_compactness() -> None:
    rows = read_csv(GEOMETRY_DIR / "internal_representation_proximity.csv")
    labels = ["GRF", "POGEMA", "SMAC"]
    envs = ["grf", "pogema", "smac"]
    values = []
    for env in envs:
        env_rows = [row for row in rows if row["env"] == env and float(row["mean_pairwise_l2"]) > 1e-6]
        values.append(statistics.median(float(row["mean_pairwise_cosine_distance"]) for row in env_rows))
    bar_chart(
        OUT_DIR / "internal-compactness.svg",
        "Within-environment compactness",
        "Median pairwise cosine distance across nondegenerate activation features.",
        labels,
        values,
        [COLORS[env] for env in envs],
        "cosine distance",
        0.1,
    )


def make_separation_bars() -> None:
    rows = read_csv(GEOMETRY_DIR / "representation_separation.csv")
    pairs = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
    values = []
    for pair in pairs:
        pair_rows = [
            row
            for row in rows
            if row["env_pair"] == pair and float(row["normalized_centroid_l2"]) < 1000
        ]
        values.append(statistics.median(float(row["normalized_centroid_l2"]) for row in pair_rows))
    bar_chart(
        OUT_DIR / "normalized-separation.svg",
        "Normalized representation separation",
        "Median centroid distance divided by pooled within-env spread; all pairs are well separated.",
        [pair.replace("_vs_", " vs ") for pair in pairs],
        values,
        [COLORS[pair] for pair in pairs],
        "normalized centroid L2",
        15,
    )


def make_asymmetric_containment() -> None:
    rows = read_csv(GEOMETRY_DIR / "asymmetric_representation_analysis.csv")
    r16 = [row for row in rows if int(row["requested_rank"]) == 16]
    directions = ["smac_to_pogema", "pogema_to_smac", "smac_to_grf", "grf_to_smac", "pogema_to_grf", "grf_to_pogema"]
    values = []
    for direction in directions:
        direction_rows = [row for row in r16 if row["direction"] == direction]
        values.append(statistics.median(float(row["target_variance_explained"]) for row in direction_rows))
    bar_chart(
        OUT_DIR / "asymmetric-containment-r16.svg",
        "Asymmetric low-rank containment",
        "Rank-16 source PCA basis explaining target variance; POGEMA contains GRF more than the reverse.",
        [direction.replace("_to_", " -> ") for direction in directions],
        values,
        [COLORS[direction] for direction in directions],
        "target variance explained",
        0.7,
    )


def make_actor_final_separation() -> None:
    rows = read_csv(GEOMETRY_DIR / "representation_separation.csv")
    features = ["actor_layer:final", "critic_layer:final", "layer_06:final"]
    pairs = ["smac_vs_pogema", "smac_vs_grf", "pogema_vs_grf"]
    series = {}
    for pair in pairs:
        vals = []
        for feature in features:
            match = next(row for row in rows if row["env_pair"] == pair and row["feature"] == feature)
            vals.append(float(match["normalized_centroid_l2"]))
        series[pair] = vals
    grouped_bar_chart(
        OUT_DIR / "final-branch-separation.svg",
        "Final and branch states make POGEMA-GRF closest",
        "Normalized centroid distance for action-facing/value-facing representations.",
        features,
        series,
        "normalized centroid L2",
        0,
        16,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_gradient_heatmap()
    make_cka_curve()
    make_internal_compactness()
    make_separation_bars()
    make_asymmetric_containment()
    make_actor_final_separation()
    methodology_schema()
    env_identity_corruption_schema()
    representation_analysis_schema()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
