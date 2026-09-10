from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ScoreFn = Callable[[frozenset[str]], float]


@dataclass(frozen=True)
class KnockoutEffect:
    module_id: str
    full_score: float
    knockout_score: float
    delta: float


@dataclass(frozen=True)
class PruningResult:
    retained_ids: tuple[str, ...]
    full_score: float
    final_score: float
    effects: tuple[KnockoutEffect, ...]
    evaluations: tuple[tuple[tuple[str, ...], float], ...]


def causal_prune(
    module_ids: tuple[str, ...] | list[str],
    score: ScoreFn,
    *,
    keep_threshold: float = 0.0,
    tolerance: float = 0.0,
    min_modules: int = 1,
) -> PruningResult:
    ordered = tuple(module_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate module ids are not allowed")
    if not ordered:
        raise ValueError("causal pruning requires at least one module")
    if min_modules < 0 or min_modules > len(ordered):
        raise ValueError("min_modules must be between 0 and the number of modules")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    order_index = {module_id: index for index, module_id in enumerate(ordered)}
    cache: dict[frozenset[str], float] = {}

    def evaluate(retained: frozenset[str]) -> float:
        if retained not in cache:
            cache[retained] = float(score(retained))
        return cache[retained]

    full = frozenset(ordered)
    full_score = evaluate(full)
    effects: list[KnockoutEffect] = []
    for module_id in ordered:
        knockout = full - {module_id}
        knockout_score = evaluate(knockout)
        effects.append(
            KnockoutEffect(
                module_id=module_id,
                full_score=full_score,
                knockout_score=knockout_score,
                delta=full_score - knockout_score,
            )
        )

    current = full
    current_score = full_score
    candidates = sorted(effects, key=lambda effect: (effect.delta, effect.module_id))
    for effect in candidates:
        if len(current) <= min_modules:
            break
        if effect.module_id not in current or effect.delta > keep_threshold:
            continue
        proposed = current - {effect.module_id}
        if len(proposed) < min_modules:
            continue
        proposed_score = evaluate(proposed)
        if proposed_score >= current_score - tolerance:
            current = proposed
            current_score = proposed_score

    retained_ids = tuple(module_id for module_id in ordered if module_id in current)
    evaluations = tuple(
        (
            tuple(module_id for module_id in ordered if module_id in retained),
            value,
        )
        for retained, value in sorted(
            cache.items(),
            key=lambda item: (
                len(item[0]),
                tuple(order_index[module_id] for module_id in ordered if module_id in item[0]),
            ),
        )
    )
    return PruningResult(
        retained_ids=retained_ids,
        full_score=full_score,
        final_score=current_score,
        effects=tuple(effects),
        evaluations=evaluations,
    )
