"""Core config loader — the public API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from agent_optimization._resolver import resolve_candidate

logger = logging.getLogger("agent_optimization")


@dataclass
class Skill:
    """A learned skill discovered during optimization."""

    name: str
    description: str
    body: str = ""


@dataclass
class OptimizationConfig:
    """Resolved optimization config.

    When not running under optimization, all fields contain the defaults
    you passed to :func:`load_config` — your agent works unchanged.
    """

    instructions: str
    model: str | None
    temperature: float | None
    skills: list[Skill] = field(default_factory=list)
    skills_dir: str | None = None
    source: str = "defaults"
    candidate_id: str | None = None

    @property
    def has_skills(self) -> bool:
        return len(self.skills) > 0 or self.skills_dir is not None

    def compose_instructions(self) -> str:
        """Return instructions with skill catalog appended (if any)."""
        if not self.skills:
            return self.instructions

        lines = [self.instructions, "", "## Available Skills"]
        for s in self.skills:
            lines.append(f"- **{s.name}**: {s.description}")
        return "\n".join(lines)


def load_config(
    *,
    default_instructions: str = "You are a helpful assistant.",
    default_model: str | None = None,
    default_temperature: float | None = None,
    default_skills_dir: str | None = None,
) -> OptimizationConfig:
    """Load optimization config with graceful fallback.

    Safe to call at module load time. When no optimization environment
    variables are present, returns your defaults unchanged.
    """
    # ── Priority 1: Candidate ID → resolver API ──────────────────────
    candidate_id = os.environ.get("AGENT_OPTIMIZATION_CANDIDATE_ID", "").strip()
    if candidate_id:
        resolved = resolve_candidate(candidate_id)
        if resolved is not None:
            return OptimizationConfig(
                instructions=resolved.get("instructions", default_instructions),
                model=resolved.get("model", default_model),
                temperature=resolved.get("temperature", default_temperature),
                skills=_parse_skills(resolved.get("skills", [])),
                skills_dir=resolved.get("skills_dir", default_skills_dir),
                source=f"api:candidate:{candidate_id}",
                candidate_id=candidate_id,
            )
        logger.warning(
            "Failed to resolve candidate %s — falling through to env/defaults",
            candidate_id,
        )

    # ── Priority 2: Config env var (inline JSON) ───────────────────
    # AGENT_OPTIMIZATION_CONFIG is set by the optimization service (first-party).
    # OPTIMIZATION_CONFIG is the non-reserved fallback used by CLI tooling.
    for env_var in ("AGENT_OPTIMIZATION_CONFIG", "OPTIMIZATION_CONFIG"):
        raw_config = os.environ.get(env_var, "").strip()
        if raw_config:
            try:
                cfg = json.loads(raw_config)
                return OptimizationConfig(
                    instructions=cfg.get("instructions", default_instructions),
                    model=cfg.get("model", default_model),
                    temperature=cfg.get("temperature", default_temperature),
                    skills=_parse_skills(cfg.get("skills", [])),
                    skills_dir=cfg.get("skills_dir", default_skills_dir),
                    source=f"env:{env_var}",
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Bad %s env var: %s", env_var, exc)

    # ── Priority 3: Defaults ─────────────────────────────────────────
    model = default_model or os.environ.get("MODEL_DEPLOYMENT_NAME")
    return OptimizationConfig(
        instructions=default_instructions,
        model=model,
        temperature=default_temperature,
        skills_dir=default_skills_dir,
        source="defaults",
    )


def _parse_skills(raw: list) -> list[Skill]:
    """Parse skills from API/env config JSON."""
    skills: list[Skill] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            skills.append(
                Skill(
                    name=item["name"],
                    description=item.get("description", ""),
                    body=item.get("body", ""),
                )
            )
    return skills
