"""Candidate config resolution via the optimization service API."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("agent_optimization")

_cache: dict[str, dict[str, Any]] = {}
_DEFAULT_SKILLS_DIR = ".agent_optimization_skills"


def resolve_candidate(candidate_id: str) -> dict[str, Any] | None:
    """Resolve a candidate's full config from the optimization service.

    Uses ``AGENT_OPTIMIZATION_RESOLVE_ENDPOINT`` env var as the base URL.
    Returns ``None`` if the endpoint is not configured or the call fails.
    """
    if candidate_id in _cache:
        return _cache[candidate_id]

    endpoint = os.environ.get("AGENT_OPTIMIZATION_RESOLVE_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        logger.debug("AGENT_OPTIMIZATION_RESOLVE_ENDPOINT not set — cannot resolve candidate")
        return None

    headers = _build_headers()

    # ── Step 1: Fetch config ─────────────────────────────────────────
    config = _api_get_json(f"{endpoint}/candidates/{candidate_id}/config", headers)
    if config is None:
        return None

    logger.info(
        "Resolved candidate %s: model=%s, instructions=%d chars, skills=%d",
        candidate_id,
        config.get("model", "?"),
        len(config.get("instructions", "")),
        len(config.get("skills", [])),
    )

    # ── Step 2: Fetch manifest and download skill files ──────────────
    skills_dir = os.environ.get("AGENT_OPTIMIZATION_SKILLS_DIR", _DEFAULT_SKILLS_DIR)
    downloaded_skills_dir = _download_skill_files(
        endpoint, candidate_id, headers, skills_dir
    )
    if downloaded_skills_dir:
        config["skills_dir"] = downloaded_skills_dir

    _cache[candidate_id] = config
    return config


def _download_skill_files(
    endpoint: str,
    candidate_id: str,
    headers: dict[str, str],
    skills_dir: str,
) -> str | None:
    """Fetch manifest, download all skill files, return the skills directory path.

    Returns ``None`` if no skill files are found or manifest fetch fails.
    """
    manifest = _api_get_json(f"{endpoint}/candidates/{candidate_id}", headers)
    if manifest is None:
        logger.debug("Could not fetch manifest for candidate %s", candidate_id)
        return None

    files = manifest.get("files", [])
    skill_files = [f for f in files if _is_skill_file(f)]
    if not skill_files:
        logger.debug("No skill files in manifest for candidate %s", candidate_id)
        return None

    logger.info(
        "Downloading %d skill file(s) for candidate %s",
        len(skill_files), candidate_id,
    )

    base_dir = pathlib.Path(skills_dir)
    for file_entry in skill_files:
        file_path = file_entry.get("path", "")
        if not file_path:
            continue

        content = _api_get_text(
            f"{endpoint}/candidates/{candidate_id}/files",
            headers,
            params={"path": file_path},
        )
        if content is None:
            logger.warning("Failed to download skill file: %s", file_path)
            continue

        # file_path is like "skills/math/SKILL.md" — write relative to base_dir
        # Strip leading "skills/" prefix so we get base_dir/math/SKILL.md
        rel_path = file_path
        if rel_path.startswith("skills/"):
            rel_path = rel_path[len("skills/"):]

        out_path = base_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        logger.info("  → %s (%d bytes)", out_path, len(content))

    return str(base_dir)


def _is_skill_file(file_entry: dict) -> bool:
    """Check if a manifest entry is a skill file."""
    path = file_entry.get("path", "")
    file_type = file_entry.get("type", "")
    return file_type == "skill" or path.startswith("skills/")


# ── HTTP helpers ─────────────────────────────────────────────────────


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    token = _get_bearer_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_get_json(url: str, headers: dict[str, str]) -> dict[str, Any] | None:
    """GET a JSON endpoint, return parsed dict or None on failure."""
    logger.debug("GET %s", url)
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.error("GET %s failed: %s", url, exc)
        return None


def _api_get_text(
    url: str, headers: dict[str, str], params: dict[str, str] | None = None
) -> str | None:
    """GET an endpoint, return response body as text or None on failure."""
    if params:
        query = "&".join(f"{k}={urllib.request.quote(v)}" for k, v in params.items())
        url = f"{url}?{query}"
    logger.debug("GET %s", url)
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        logger.error("GET %s failed: %s", url, exc)
        return None


def _get_bearer_token() -> str | None:
    """Acquire a bearer token for the resolver API.

    Uses ``azure-identity`` if available; returns ``None`` otherwise.
    This keeps azure-identity as an optional dependency.
    """
    try:
        from azure.identity import DefaultAzureCredential

        cred = DefaultAzureCredential()
        token = cred.get_token("https://ml.azure.com/.default")
        return token.token
    except Exception:  # noqa: BLE001
        return None
