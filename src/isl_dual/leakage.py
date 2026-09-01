from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class SecretBundle:
    expert_artifact: Any = None
    expert_trajectory: Any = None
    curated_skill: Any = None
    benchmark_hidden_reward: Any = None


def _contains(payload: Any, secret: Any) -> bool:
    if secret is None:
        return False
    if payload is secret:
        return True
    if isinstance(payload, str) and isinstance(secret, str):
        return bool(secret) and secret in payload
    if isinstance(payload, dict):
        return any(_contains(k, secret) or _contains(v, secret) for k, v in payload.items())
    if isinstance(payload, (list, tuple, set)):
        return any(_contains(item, secret) for item in payload)
    return False


def assert_inverse_input(payload: Any, secrets: SecretBundle) -> None:
    assert not _contains(payload, secrets.expert_trajectory), "expert trajectory leaked into inverse input"
    assert not _contains(payload, secrets.curated_skill), "curated skill leaked into inverse input"


def assert_forward_input(payload: Any, secrets: SecretBundle) -> None:
    assert not _contains(payload, secrets.expert_artifact), "expert artifact leaked into forward input"
    assert not _contains(payload, secrets.expert_trajectory), "expert trajectory leaked into forward input"
    assert not _contains(payload, secrets.curated_skill), "curated skill leaked into forward input"


def assert_deployment_input(payload: Any, secrets: SecretBundle) -> None:
    assert not _contains(payload, secrets.expert_artifact), "expert artifact leaked into deployment input"
    assert not _contains(payload, secrets.benchmark_hidden_reward), "hidden reward leaked into deployment input"

