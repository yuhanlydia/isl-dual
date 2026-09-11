from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import MutableMapping


def _valid_codex_auth_bytes(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Codex auth JSON must contain an object")
        tokens = payload.get("tokens")
        valid = (
            payload.get("auth_mode") in (None, "chatgpt")
            and not payload.get("OPENAI_API_KEY")
            and isinstance(tokens, dict)
            and bool(tokens.get("access_token"))
            and bool(tokens.get("refresh_token"))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Could not read valid Codex OAuth credentials from {path}") from exc
    if not valid:
        raise RuntimeError(f"Could not read valid Codex OAuth credentials from {path}")
    return raw


def _codex_account_identity(raw: bytes) -> str | None:
    payload = json.loads(raw)
    tokens = payload.get("tokens", {})
    account_id = tokens.get("account_id")
    if account_id:
        return f"account:{account_id}"
    id_token = tokens.get("id_token")
    if not isinstance(id_token, str):
        return None
    try:
        encoded = id_token.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded))
    except (IndexError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict):
        return None
    namespaced = claims.get("https://api.openai.com/auth")
    if isinstance(namespaced, dict):
        for claim in ("chatgpt_account_id", "chatgpt_user_id"):
            value = namespaced.get(claim)
            if value:
                return f"{claim}:{value}"
    for claim in ("chatgpt_account_id", "sub", "email"):
        value = claims.get(claim)
        if value:
            return f"{claim}:{value}"
    return None


class CodexOAuthSession:
    """Serialize OAuth use and synchronize refreshes through a private copy."""

    def __init__(self, auth_path: Path) -> None:
        self.auth_path = Path(auth_path).expanduser().resolve()
        self.staging_path: Path
        self._lock_file = None
        self._tempdir = None
        self._account_id: str | None = None
        self._initial_bytes: bytes | None = None

    def __enter__(self):
        digest = hashlib.sha256(str(self.auth_path).encode()).hexdigest()[:16]
        lock_path = Path(tempfile.gettempdir()) / f"isl-codex-auth-{digest}.lock"
        self._lock_file = lock_path.open("a+b")
        os.chmod(lock_path, 0o600)
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        raw = _valid_codex_auth_bytes(self.auth_path)
        self._initial_bytes = raw
        self._account_id = _codex_account_identity(raw)
        self._tempdir = tempfile.TemporaryDirectory(prefix="isl-codex-auth-")
        self.staging_path = Path(self._tempdir.name) / "auth.json"
        self.staging_path.write_bytes(raw)
        os.chmod(self.staging_path, 0o600)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            staged = _valid_codex_auth_bytes(self.staging_path)
            current = _valid_codex_auth_bytes(self.auth_path)
            if staged != current:
                if current != self._initial_bytes:
                    raise RuntimeError(
                        "Host Codex OAuth credentials changed during the run; refusing writeback"
                    )
                staged_account = _codex_account_identity(staged)
                if self._account_id is None:
                    raise RuntimeError(
                        "Codex OAuth refresh has no stable account identity; refusing host credential writeback"
                    )
                if staged_account != self._account_id:
                    raise RuntimeError("Codex OAuth identity changed; refusing host credential writeback")
                with tempfile.NamedTemporaryFile(
                    dir=self.auth_path.parent,
                    prefix=".isl-auth-",
                    delete=False,
                ) as output:
                    output.write(staged)
                    replacement = Path(output.name)
                os.chmod(replacement, 0o600)
                os.replace(replacement, self.auth_path)
        finally:
            if self._tempdir is not None:
                self._tempdir.cleanup()
            if self._lock_file is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()


class DockerAuthTransportSubprocess:
    """Proxy Docker calls to limit OAuth visibility to the agent phase."""

    def __init__(self, delegate, auth_path: Path) -> None:
        self._delegate = delegate
        self._auth_path = Path(auth_path).expanduser().resolve()
        self._active: set[str] = set()

    @staticmethod
    def _container_name(command) -> str | None:
        try:
            index = command.index("--name")
            return str(command[index + 1])
        except (ValueError, IndexError):
            return None

    def _scrub(self, container: str) -> None:
        if container not in self._active:
            return
        stopped = self._delegate.run(
            [
                "docker", "exec", container, "sh", "-c",
                "pkill -f '[c]odex' || true; "
                "i=0; while pgrep -f '[c]odex' >/dev/null && [ $i -lt 100 ]; do "
                "sleep 0.1; i=$((i+1)); done; ! pgrep -f '[c]odex' >/dev/null",
            ],
            capture_output=True,
        )
        if stopped.returncode != 0:
            raise RuntimeError("Could not stop Codex before OAuth credential retrieval")
        copied = self._delegate.run(
            ["docker", "cp", f"{container}:/root/.codex/auth.json", str(self._auth_path)],
            capture_output=True,
        )
        if copied.returncode != 0:
            raise RuntimeError("Could not retrieve refreshed Codex OAuth credentials")
        os.chmod(self._auth_path, 0o600)
        removed = self._delegate.run(
            ["docker", "exec", container, "rm", "-f", "/root/.codex/auth.json"],
            capture_output=True,
        )
        if removed.returncode != 0:
            raise RuntimeError("Could not remove Codex OAuth credentials before verifier")
        self._active.discard(container)

    def _force_remove_after_scrub_failure(self, container: str, exc: Exception) -> None:
        removed = self._delegate.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
        )
        self._active.discard(container)
        status = "force-removed" if removed.returncode == 0 else "could not be force-removed"
        raise RuntimeError(
            f"{exc}; container {container!r} was {status}; Codex login may require re-authentication"
        ) from exc

    def run(self, command, *args, **kwargs):
        is_docker = (
            isinstance(command, (list, tuple))
            and len(command) >= 2
            and command[0] == "docker"
        )
        if is_docker and command[1] == "exec" and len(command) >= 5:
            if command[3:5] == ["bash", "/tests/test.sh"]:
                container = str(command[2])
                try:
                    self._scrub(container)
                except RuntimeError as exc:
                    self._force_remove_after_scrub_failure(container, exc)
        elif is_docker and command[1] == "rm" and len(command) >= 4:
            container = str(command[-1])
            try:
                self._scrub(container)
            except RuntimeError as exc:
                self._force_remove_after_scrub_failure(container, exc)

        result = self._delegate.run(command, *args, **kwargs)
        if is_docker and command[1] == "run" and result.returncode == 0:
            container = self._container_name(command)
            if container:
                process_tools = self._delegate.run(
                    [
                        "docker", "exec", container, "sh", "-c",
                        "command -v pkill >/dev/null && command -v pgrep >/dev/null || "
                        "(apt-get update -qq && apt-get install -y --no-install-recommends procps)",
                    ],
                    capture_output=True,
                )
                if process_tools.returncode != 0:
                    self._delegate.run(
                        ["docker", "rm", "-f", container],
                        capture_output=True,
                    )
                    raise RuntimeError("Could not prepare process-control tools for Codex OAuth")
                mkdir = self._delegate.run(
                    ["docker", "exec", container, "mkdir", "-p", "/root/.codex"],
                    capture_output=True,
                )
                copied = self._delegate.run(
                    ["docker", "cp", str(self._auth_path), f"{container}:/root/.codex/auth.json"],
                    capture_output=True,
                )
                if mkdir.returncode != 0 or copied.returncode != 0:
                    self._delegate.run(
                        ["docker", "rm", "-f", container],
                        capture_output=True,
                    )
                    raise RuntimeError("Could not inject Codex OAuth credentials into benchmark container")
                self._active.add(container)
        return result

    def close(self) -> None:
        for container in list(self._active):
            try:
                self._scrub(container)
            except RuntimeError as exc:
                self._force_remove_after_scrub_failure(container, exc)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def effective_max_workers(requested: int, auth_mode: str) -> int:
    return 1 if auth_mode == "oauth" else requested


def configure_codex_auth(
    agent: dict,
    *,
    environ: MutableMapping[str, str],
    auth_path: Path,
) -> str:
    """Configure SkillLearnBench's Codex agent for API-key or OAuth auth."""
    if environ.get("OPENAI_API_KEY", "").strip():
        return "api_key"

    auth_path = Path(auth_path).expanduser().resolve()
    if not auth_path.is_file():
        raise RuntimeError(
            "Codex requires OPENAI_API_KEY or a Codex OAuth login at "
            f"{auth_path}"
        )

    _valid_codex_auth_bytes(auth_path)

    agent["env"] = []
    agent["setup"] = None
    trajectory_env = dict(agent.get("trajectory_env", {}))
    trajectory_env["CODEX_HOME"] = "/root/.codex"
    agent["trajectory_env"] = trajectory_env
    return "oauth"


def _load_upstream(root: Path):
    root = root.resolve()
    if not (root / "evaluate_skills.py").is_file():
        raise FileNotFoundError(f"evaluate_skills.py is missing under {root}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("evaluate_skills")
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Thin bridge to SkillLearnBench.hyper_eval without its unconditional Anthropic-only CLI guard."
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill-path", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--build-workers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args(argv)

    root = args.benchmark_root.resolve()
    upstream = _load_upstream(root)
    upstream._load_dotenv()

    skill_path = None if args.skill_path.lower() == "none" else Path(args.skill_path).resolve()
    skill_paths = [skill_path]
    task_root = root / "tasks"

    oauth_auth_path: Path | None = None
    agent_to_restore = None
    original_agent = None
    eval_runner = None
    original_subprocess = None
    transport = None
    oauth_session = None
    oauth_entered = False
    try:
        if not args.dry_run and args.agent == "codex":
            configured_path = os.environ.get("CODEX_AUTH_JSON")
            auth_path = (
                Path(configured_path)
                if configured_path
                else Path.home() / ".codex" / "auth.json"
            )
            agent_to_restore = upstream.get_agent("codex")
            original_agent = copy.deepcopy(agent_to_restore)
            auth_mode = configure_codex_auth(
                agent_to_restore,
                environ=os.environ,
                auth_path=auth_path,
            )
            args.max_workers = effective_max_workers(args.max_workers, auth_mode)
            if auth_mode == "oauth":
                oauth_auth_path = auth_path.expanduser().resolve()
        if not args.dry_run:
            upstream._require_docker()
            upstream._validate_api_keys(
                agent_id=args.agent,
                task_ids=list(args.tasks),
                task_root=task_root,
                need_agent_keys=True,
                need_judge_key=not args.skip_metrics,
            )
            upstream._ensure_skill_paths(skill_paths, list(args.tasks))

        if oauth_auth_path is not None:
            oauth_session = CodexOAuthSession(oauth_auth_path)
            session = oauth_session.__enter__()
            oauth_entered = True
            eval_runner = importlib.import_module("core.eval_runner")
            original_subprocess = eval_runner.subprocess
            transport = DockerAuthTransportSubprocess(
                original_subprocess,
                session.staging_path,
            )
            eval_runner.subprocess = transport

        return int(upstream.hyper_eval(
            list(args.tasks),
            task_root=task_root,
            agent_id=args.agent,
            model=args.model,
            skill_paths=skill_paths,
            repeats=max(1, args.repeats),
            max_steps=max(1, args.max_steps),
            max_workers=max(1, args.max_workers),
            build_workers=max(1, args.build_workers),
            remove_images=False,
            record=True,
            dry_run=args.dry_run,
            config_path=None,
            trials_dir=args.trials_dir.resolve(),
        ))
    finally:
        try:
            if transport is not None:
                transport.close()
        finally:
            try:
                if eval_runner is not None:
                    eval_runner.subprocess = original_subprocess
            finally:
                try:
                    if oauth_session is not None and oauth_entered:
                        oauth_session.__exit__(*sys.exc_info())
                finally:
                    if agent_to_restore is not None and original_agent is not None:
                        agent_to_restore.clear()
                        agent_to_restore.update(original_agent)


if __name__ == "__main__":
    raise SystemExit(main())
