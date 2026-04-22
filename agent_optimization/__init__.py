"""Agent Optimization — Config loader for optimization-ready hosted agents.

One import, one call::

    from agent_optimization import load_config

    config = load_config(default_instructions="You are a helpful assistant.")
    # config.instructions  — optimized or default
    # config.model         — optimized or default
    # config.temperature   — optimized or default
    # config.skills        — learned skills (empty if none)
    # config.source        — "api:candidate:abc", "env:config", or "defaults"

Resolution order:
    1. AGENT_OPTIMIZATION_CANDIDATE_ID env → resolver API → full config + skills
    2. AGENT_OPTIMIZATION_CONFIG env var   → inline JSON fallback
    3. Defaults              → your hardcoded values (agent works normally)
"""

from agent_optimization._config import OptimizationConfig, Skill, load_config

__all__ = ["OptimizationConfig", "Skill", "load_config"]
__version__ = "0.1.0"
