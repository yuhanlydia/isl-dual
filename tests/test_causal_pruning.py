from __future__ import annotations

from isl_dual.causal_pruning import causal_prune


def test_causal_prune_keeps_helpful_and_removes_harmful_or_neutral_modules() -> None:
    modules = ("helpful", "neutral", "harmful")

    def score(retained: frozenset[str]) -> float:
        value = 0.5
        if "helpful" in retained:
            value += 0.3
        if "harmful" in retained:
            value -= 0.2
        return value

    result = causal_prune(modules, score, min_modules=1)

    assert result.full_score == 0.6
    assert result.final_score == 0.8
    assert result.retained_ids == ("helpful",)
    effects = {effect.module_id: effect.delta for effect in result.effects}
    assert effects["helpful"] > 0
    assert effects["neutral"] == 0
    assert effects["harmful"] < 0


def test_causal_prune_uses_current_score_for_conservative_greedy_deletions() -> None:
    modules = ("a", "b", "c")
    table = {
        frozenset({"a", "b", "c"}): 0.90,
        frozenset({"b", "c"}): 0.70,  # a is clearly helpful
        frozenset({"a", "c"}): 0.90,  # b initially neutral
        frozenset({"a", "b"}): 0.89,  # c slightly helpful but within tolerance
        frozenset({"a", "c"}): 0.90,
        frozenset({"a"}): 0.86,       # after b is removed, removing c would exceed tolerance
    }

    result = causal_prune(
        modules,
        lambda retained: table[retained],
        tolerance=0.02,
        min_modules=1,
    )

    assert result.retained_ids == ("a", "c")
    assert result.final_score == 0.90


def test_causal_prune_is_deterministic_on_ties() -> None:
    modules = ("z", "a", "m")
    seen: list[tuple[str, ...]] = []

    def score(retained: frozenset[str]) -> float:
        seen.append(tuple(sorted(retained)))
        return 1.0

    first = causal_prune(modules, score, min_modules=1)
    second = causal_prune(modules, score, min_modules=1)

    assert first.retained_ids == second.retained_ids
    assert first.retained_ids == ("z",)


def test_causal_prune_respects_min_modules() -> None:
    modules = ("a", "b", "c")

    result = causal_prune(modules, lambda retained: float(len(retained) == 0), min_modules=2)

    assert len(result.retained_ids) == 2
    assert result.retained_ids == ("b", "c")


def test_causal_prune_rejects_duplicate_module_ids() -> None:
    try:
        causal_prune(("a", "a"), lambda retained: 1.0)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("duplicate module ids should fail")
