from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from isl_dual.skilllearn_bridge import (
    CodexOAuthSession,
    DockerAuthTransportSubprocess,
    configure_codex_auth,
    effective_max_workers,
)


def _oauth_file(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": "id-secret",
                    "access_token": "access-secret",
                    "refresh_token": "refresh-secret",
                    "account_id": "account-secret",
                },
            }
        )
    )
    return path


def _jwt(payload: dict[str, str]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_codex_oauth_fallback_configures_ephemeral_container_auth(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }
    environ: dict[str, str] = {}

    mode = configure_codex_auth(agent, environ=environ, auth_path=auth_path)

    assert mode == "oauth"
    assert agent["env"] == []
    assert agent["setup"] is None
    assert agent["trajectory_env"]["CODEX_HOME"] == "/root/.codex"
    assert environ == {}


def test_codex_api_key_path_remains_unchanged(tmp_path: Path) -> None:
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }
    original = json.loads(json.dumps(agent))
    environ = {"OPENAI_API_KEY": "sk-test"}

    mode = configure_codex_auth(
        agent,
        environ=environ,
        auth_path=tmp_path / "missing.json",
    )

    assert mode == "api_key"
    assert agent == original
    assert environ == {"OPENAI_API_KEY": "sk-test"}


def test_oauth_is_copied_for_agent_then_removed_before_verifier(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")

    class Recorder:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def run(self, command: list[str], **kwargs):
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0)

    recorder = Recorder()
    transported = DockerAuthTransportSubprocess(recorder, auth_path)

    transported.run(["docker", "build", "-t", "image", "."])
    transported.run(["docker", "run", "-d", "--name", "trial", "image", "sleep", "3600"])
    transported.run(["docker", "exec", "trial", "bash", "/tests/test.sh"])

    assert recorder.commands[0] == ["docker", "build", "-t", "image", "."]
    assert recorder.commands[1] == [
        "docker", "run", "-d", "--name", "trial", "image", "sleep", "3600",
    ]
    verifier_index = recorder.commands.index(
        ["docker", "exec", "trial", "bash", "/tests/test.sh"]
    )
    remove_index = recorder.commands.index(
        ["docker", "exec", "trial", "rm", "-f", "/root/.codex/auth.json"]
    )
    assert remove_index < verifier_index
    assert any(
        command[:3] == ["docker", "cp", str(auth_path.resolve())]
        and command[3] == "trial:/root/.codex/auth.json"
        for command in recorder.commands
    )
    assert "access-secret" not in " ".join(
        part for command in recorder.commands for part in command
    )


def test_oauth_session_syncs_refreshed_credentials_back_atomically(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    refreshed = json.loads(auth_path.read_text())
    refreshed["tokens"]["refresh_token"] = "rotated-secret"

    with CodexOAuthSession(auth_path) as session:
        session.staging_path.write_text(json.dumps(refreshed))

    assert json.loads(auth_path.read_text()) == refreshed
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_oauth_transport_never_runs_verifier_when_scrub_fails(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")

    class FailingCopyOutRecorder:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def run(self, command: list[str], **kwargs):
            self.commands.append(command)
            failed = command[:2] == ["docker", "cp"] and str(command[2]).startswith("trial:")
            return subprocess.CompletedProcess(command, 1 if failed else 0)

    recorder = FailingCopyOutRecorder()
    transported = DockerAuthTransportSubprocess(recorder, auth_path)
    transported.run(["docker", "run", "-d", "--name", "trial", "image", "sleep", "3600"])

    with pytest.raises(RuntimeError, match="retrieve.*OAuth"):
        transported.run(["docker", "exec", "trial", "bash", "/tests/test.sh"])

    assert ["docker", "exec", "trial", "bash", "/tests/test.sh"] not in recorder.commands
    assert ["docker", "rm", "-f", "trial"] in recorder.commands


def test_oauth_is_not_injected_when_process_tools_cannot_be_prepared(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")

    class MissingProcpsRecorder:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def run(self, command: list[str], **kwargs):
            self.commands.append(command)
            is_procps_setup = command[:4] == ["docker", "exec", "trial", "sh"]
            return subprocess.CompletedProcess(command, 1 if is_procps_setup else 0)

    recorder = MissingProcpsRecorder()
    transported = DockerAuthTransportSubprocess(recorder, auth_path)

    with pytest.raises(RuntimeError, match="process-control tools"):
        transported.run(["docker", "run", "-d", "--name", "trial", "image", "sleep", "3600"])

    assert not any(command[:2] == ["docker", "cp"] for command in recorder.commands)


def test_oauth_session_refuses_cross_account_writeback(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    original = auth_path.read_bytes()

    with pytest.raises(RuntimeError, match="identity changed"):
        with CodexOAuthSession(auth_path) as session:
            changed = json.loads(session.staging_path.read_text())
            changed["tokens"]["account_id"] = "different-account"
            session.staging_path.write_text(json.dumps(changed))

    assert auth_path.read_bytes() == original


def test_oauth_serializes_trials_while_api_key_keeps_requested_workers() -> None:
    assert effective_max_workers(3, "oauth") == 1
    assert effective_max_workers(3, "api_key") == 3


def test_codex_auth_rejects_missing_api_key_and_oauth_file(tmp_path: Path) -> None:
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY.*Codex OAuth"):
        configure_codex_auth(
            agent,
            environ={},
            auth_path=tmp_path / "missing.json",
        )


def test_codex_auth_rejects_malformed_oauth_file(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"auth_mode": "chatgpt", "tokens": {}}')
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }

    with pytest.raises(RuntimeError, match="valid Codex OAuth"):
        configure_codex_auth(agent, environ={}, auth_path=auth_path)


def test_codex_auth_accepts_valid_oauth_without_optional_account_id(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    payload = json.loads(auth_path.read_text())
    payload["tokens"].pop("account_id")
    payload["tokens"]["id_token"] = _jwt({"sub": "stable-user"})
    auth_path.write_text(json.dumps(payload))
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }

    assert configure_codex_auth(agent, environ={}, auth_path=auth_path) == "oauth"


def test_codex_auth_accepts_legacy_oauth_with_omitted_auth_mode(tmp_path: Path) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    payload = json.loads(auth_path.read_text())
    payload.pop("auth_mode")
    auth_path.write_text(json.dumps(payload))
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
        "run": "codex exec --json",
    }

    assert configure_codex_auth(agent, environ={}, auth_path=auth_path) == "oauth"


def test_oauth_session_syncs_refresh_without_account_id_using_id_token_subject(
    tmp_path: Path,
) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    original = json.loads(auth_path.read_text())
    original["tokens"].pop("account_id")
    original["tokens"]["id_token"] = _jwt(
        {
            "email": "person@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "workspace-a",
                "chatgpt_user_id": "stable-user",
            },
        }
    )
    auth_path.write_text(json.dumps(original))

    with CodexOAuthSession(auth_path) as session:
        refreshed = json.loads(session.staging_path.read_text())
        refreshed["tokens"]["access_token"] = "new-access"
        refreshed["tokens"]["refresh_token"] = "new-refresh"
        refreshed["tokens"]["id_token"] = _jwt(
            {
                "email": "person@example.com",
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "workspace-a",
                    "chatgpt_user_id": "stable-user",
                },
            }
        )
        session.staging_path.write_text(json.dumps(refreshed))

    assert json.loads(auth_path.read_text()) == refreshed


def test_oauth_session_rejects_same_email_refresh_for_different_workspace(
    tmp_path: Path,
) -> None:
    auth_path = _oauth_file(tmp_path / "auth.json")
    original = json.loads(auth_path.read_text())
    original["tokens"].pop("account_id")
    original["tokens"]["id_token"] = _jwt(
        {
            "email": "person@example.com",
            "https://api.openai.com/auth": {"chatgpt_account_id": "workspace-a"},
        }
    )
    auth_path.write_text(json.dumps(original))

    with pytest.raises(RuntimeError, match="identity changed"):
        with CodexOAuthSession(auth_path) as session:
            changed = json.loads(session.staging_path.read_text())
            changed["tokens"]["refresh_token"] = "other-workspace-refresh"
            changed["tokens"]["id_token"] = _jwt(
                {
                    "email": "person@example.com",
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "workspace-b"
                    },
                }
            )
            session.staging_path.write_text(json.dumps(changed))
