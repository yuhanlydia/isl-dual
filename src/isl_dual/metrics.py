from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def entropy(posterior: Mapping[str, float]) -> float:
    return -sum(p * math.log(p) for p in posterior.values() if p > 0)


def skill_lift(isl_reward: float, no_skill_reward: float) -> float:
    return isl_reward - no_skill_reward


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman requires equal sequences of length >= 2")
    x, y = _ranks(left), _ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def spurious_skill_rejection_rate(
    artifact_scores: Mapping[str, float], deployment_scores: Mapping[str, float],
    q0: Mapping[str, float], q2: Mapping[str, float], tau_artifact: float, tau_transfer: float,
) -> float | None:
    spurious = [k for k in artifact_scores if artifact_scores[k] > tau_artifact and deployment_scores[k] < tau_transfer]
    if not spurious:
        return None
    return sum(q2.get(k, 0.0) < q0.get(k, 0.0) for k in spurious) / len(spurious)

