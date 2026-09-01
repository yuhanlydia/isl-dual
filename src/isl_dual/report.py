from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .metrics import entropy, skill_lift, spearman, spurious_skill_rejection_rate


def scientific_report(
    *, no_skill_reward: float, isl_reward: float,
    artifact_scores: Mapping[str, float], forward_scores: Mapping[str, float],
    deployment_scores: Mapping[str, float], q0: Mapping[str, float], q2: Mapping[str, float],
    tau_artifact: float, tau_transfer: float,
) -> dict[str, Any]:
    common = sorted(set(artifact_scores) & set(forward_scores) & set(deployment_scores))
    if len(common) < 2:
        raise ValueError("candidate-level correlations require at least two common graphs")
    rho_forward = spearman([forward_scores[k] for k in common], [deployment_scores[k] for k in common])
    rho_static = spearman([artifact_scores[k] for k in common], [deployment_scores[k] for k in common])
    return {
        "skill_lift": skill_lift(isl_reward, no_skill_reward),
        "rho_forward": rho_forward,
        "rho_static": rho_static,
        "forward_correlation_advantage": rho_forward - rho_static,
        "spurious_skill_rejection_rate": spurious_skill_rejection_rate(
            artifact_scores, deployment_scores, q0, q2, tau_artifact, tau_transfer,
        ),
        "posterior_entropy_q0": entropy(q0),
        "posterior_entropy_q2": entropy(q2),
        "posterior_entropy_reduction": entropy(q0) - entropy(q2),
    }


def go_gate(baseline_rewards: Mapping[str, float]) -> dict[str, Any]:
    required = {"B1", "B3", "B4", "B6"}
    missing = required - set(baseline_rewards)
    if missing:
        raise ValueError(f"GO gate missing baselines: {sorted(missing)}")
    forward_beats_static = baseline_rewards["B4"] > baseline_rewards["B3"]
    dual_beats_direct = baseline_rewards["B6"] > baseline_rewards["B1"]
    flat = max(baseline_rewards[k] for k in ("B3", "B4", "B6")) - min(baseline_rewards[k] for k in ("B3", "B4", "B6")) < 1e-9
    return {
        "B4_gt_B3": forward_beats_static,
        "B6_gt_B1": dual_beats_direct,
        "B3_B4_B6_flat": flat,
        "go": forward_beats_static and dual_beats_direct and not flat,
    }
