"""Eligibility audit for actor and critic CLT circuit tracing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def audit_clt_suite(
    actor_metrics: Mapping[str, float],
    critic_metrics: Mapping[str, float],
    replacement_metrics: Mapping[str, float],
    *,
    environments: Sequence[str],
    num_layers: int,
    maximum_normalized_mse: float,
    maximum_dead_feature_fraction: float,
    minimum_l0: float,
    maximum_l0: float,
    maximum_actor_kl: float,
    minimum_action_agreement: float,
    maximum_critic_value_mae: float,
) -> dict[str, Any]:
    """Return a complete pass/fail record; missing metrics are failures."""

    failures: list[dict[str, Any]] = []
    checked: dict[str, float] = {}

    def check(name: str, value: float | None, relation: str, threshold: float) -> None:
        if value is None:
            failures.append({"metric": name, "reason": "missing"})
            return
        numeric = float(value)
        checked[name] = numeric
        passed = numeric <= threshold if relation == "maximum" else numeric >= threshold
        if not passed:
            failures.append(
                {
                    "metric": name,
                    "value": numeric,
                    "required": f"{relation} {threshold}",
                }
            )

    groups = ("all", *environments)
    for branch, metrics in (("actor", actor_metrics), ("critic", critic_metrics)):
        for group in groups:
            prefix = "validation/" if group == "all" else f"validation/environment/{group}/"
            for layer in range(num_layers):
                layer_prefix = f"{prefix}layer_{layer:02d}"
                check(
                    f"{branch}/{group}/layer_{layer:02d}/normalized_mse",
                    metrics.get(f"{layer_prefix}/normalized_mse"),
                    "maximum",
                    maximum_normalized_mse,
                )
                check(
                    f"{branch}/{group}/layer_{layer:02d}/dead_feature_fraction",
                    metrics.get(f"{layer_prefix}/dead_feature_fraction"),
                    "maximum",
                    maximum_dead_feature_fraction,
                )
                check(
                    f"{branch}/{group}/layer_{layer:02d}/l0_minimum",
                    metrics.get(f"{layer_prefix}/l0"),
                    "minimum",
                    minimum_l0,
                )
                check(
                    f"{branch}/{group}/layer_{layer:02d}/l0_maximum",
                    metrics.get(f"{layer_prefix}/l0"),
                    "maximum",
                    maximum_l0,
                )
    for group in groups:
        check(
            f"replacement/{group}/actor_kl",
            replacement_metrics.get(f"{group}/actor_kl"),
            "maximum",
            maximum_actor_kl,
        )
        check(
            f"replacement/{group}/action_agreement",
            replacement_metrics.get(f"{group}/action_agreement"),
            "minimum",
            minimum_action_agreement,
        )
        check(
            f"replacement/{group}/critic_value_mae",
            replacement_metrics.get(f"{group}/critic_value_mae"),
            "maximum",
            maximum_critic_value_mae,
        )
    return {
        "status": "passed" if not failures else "failed",
        "eligible_for_graph_interpretation": not failures,
        "checked_metrics": checked,
        "failures": failures,
    }
