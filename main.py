"""Optimization-ready hosted agent (minimal).

Zero agentserver/agent_framework deps — FastAPI + OpenAI Responses API.
Speaks the Foundry contract: POST /responses, GET /liveness, GET /readiness.

Supports agent optimization out of the box via load_config().
Discovers file-based skills (agentskills.io format) with progressive disclosure:
  1. Startup — skill name + description in system prompt
  2. On demand — model calls load_skill tool → full SKILL.md body
  3. Deep dive — model calls read_skill_file → scripts/, references/, assets/

Normal operation (no optimization):
    Uses the defaults you specify below — behaves like a regular chat agent.

During optimization:
    The optimization service injects AGENT_OPTIMIZATION_CONFIG env vars.
    load_config() picks them up and returns the candidate's config.
"""

import json
import os
import re
import uuid
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from agent_optimization import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimize-agent")

# ── Skills discovery (agentskills.io) ─────────────────────────────────────────

def discover_skills(skills_dir="skills"):
    """Scan dir for SKILL.md files → list of {name, description, body, path}."""
    skills = []
    root = Path(skills_dir)
    if not root.is_dir():
        return skills
    for skill_file in root.rglob("SKILL.md"):
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
        if not m:
            continue
        frontmatter, body = m.group(1), m.group(2).strip()
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", frontmatter, re.MULTILINE))
        if meta.get("name"):
            skills.append({
                "name": meta["name"],
                "description": meta.get("description", ""),
                "body": body,
                "path": str(skill_file.parent),
            })
            logger.info("Discovered skill: %s (%s)", meta["name"], skill_file.parent)
    return skills


def build_system_prompt(instructions, skills):
    """Append skill catalog (name + description only) per progressive disclosure."""
    if not skills:
        return instructions
    lines = [instructions, "", "## Available Skills",
             "Call `load_skill` to activate a skill before using it."]
    for s in skills:
        lines.append(f"- **{s['name']}**: {s['description']}")
    return "\n".join(lines)


SKILL_TOOLS = [
    {
        "type": "function",
        "name": "load_skill",
        "description": "Activate a skill by loading its full instructions.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Skill name"}},
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "read_skill_file",
        "description": "Read a file from a skill's directory (scripts/, references/, assets/).",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name"},
                "file_path": {"type": "string", "description": "Relative path within the skill dir"},
            },
            "required": ["skill_name", "file_path"],
        },
    },
]


def handle_tool_call(name, args, skill_map):
    """Execute a skill tool call, return the result string."""
    if name == "load_skill":
        skill = skill_map.get(args.get("name", ""))
        if not skill:
            return f"Unknown skill: {args.get('name')}. Available: {', '.join(skill_map)}"
        return skill["body"]
    if name == "read_skill_file":
        skill = skill_map.get(args.get("skill_name", ""))
        if not skill:
            return f"Unknown skill: {args.get('skill_name')}"
        target = Path(skill["path"]) / args.get("file_path", "")
        if not target.is_file():
            return f"File not found: {args.get('file_path')}"
        return target.read_text(encoding="utf-8", errors="replace")[:10000]
    return f"Unknown tool: {name}"


# ── Config ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant.

Be concise, accurate, and helpful."""

config = load_config(
    default_instructions=SYSTEM_PROMPT,
    default_model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
    default_skills_dir="skills",
)

local_skills = discover_skills(config.skills_dir or "skills")
all_instructions = config.compose_instructions()
if local_skills and not config.skills:
    all_instructions = build_system_prompt(all_instructions, local_skills)

MODEL = config.model or os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5-mini")
SKILL_MAP = {s["name"]: s for s in local_skills}
TOOLS = SKILL_TOOLS if local_skills else []

logger.info("Config loaded (source=%s, model=%s, skills=%d, tools=%d)",
            config.source, MODEL, len(local_skills), len(TOOLS))

# ── OpenAI client (lazy — created on first request) ──────────────────────────
# Route model calls through a project endpoint so the platform auto-grants
# the necessary RBAC. FOUNDRY_PROJECT_ENDPOINT is injected by the platform at
# runtime; AZURE_OPENAI_PROJECT_ENDPOINT allows using a different project for
# model calls (e.g., when deploy project differs from model project).
MODEL_ENDPOINT = (
    os.environ.get("AZURE_OPENAI_PROJECT_ENDPOINT")
    or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    or os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
)
API_VERSION = "2025-11-15-preview"

_oai_client = None

def get_oai():
    global _oai_client
    if _oai_client is not None:
        return _oai_client
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    _credential = DefaultAzureCredential()
    _token_provider = get_bearer_token_provider(
        _credential, "https://ai.azure.com/.default"
    )
    _oai_client = AzureOpenAI(
        azure_endpoint=MODEL_ENDPOINT,
        azure_ad_token_provider=_token_provider,
        api_version=API_VERSION,
    )
    logger.info("OAI client: endpoint=%s api_version=%s", MODEL_ENDPOINT, API_VERSION)
    return _oai_client

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/liveness")
async def liveness():
    return {"status": "alive"}

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

MAX_TOOL_ROUNDS = 5

@app.post("/responses")
@app.post("/runs")
async def responses(request: Request):
    """Handle Foundry invoke via OpenAI Responses API."""
    body = await request.json()

    raw_input = body.get("input", "")
    instructions = body.get("instructions", all_instructions)
    model = body.get("model", MODEL)

    logger.info("Request: model=%s body_keys=%s", model, list(body.keys()))

    # Use platform-provided response_id from metadata
    metadata = body.get("metadata", {}) or {}
    response_id = (metadata.get("response_id") if isinstance(metadata, dict) else None)
    if not response_id:
        response_id = f"caresp_{uuid.uuid4().hex}{uuid.uuid4().hex[:18]}"

    oai = get_oai()

    # Build tools list for Responses API
    resp_tools = None
    if TOOLS:
        resp_tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in TOOLS if t["type"] == "function"]

    try:
        resp = oai.responses.create(
            model=model,
            instructions=instructions,
            input=raw_input,
            tools=resp_tools or [],
        )
    except Exception as e:
        logger.error("AOAI Responses call failed: %s", e)
        return JSONResponse(content={
            "id": response_id, "object": "response",
            "created_at": int(time.time()), "status": "completed",
            "model": model,
            "output": [{"type": "message", "id": f"msg_{uuid.uuid4().hex}{uuid.uuid4().hex[:14]}",
                "status": "completed", "role": "assistant",
                "content": [{"type": "output_text",
                    "text": f"AOAI ERROR: {type(e).__name__}: {e}"}]}],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        })

    # Tool-calling loop
    for _round in range(MAX_TOOL_ROUNDS):
        tool_calls = [item for item in resp.output if item.type == "function_call"]
        if not tool_calls:
            break
        logger.info("Tool round %d: %d calls", _round + 1, len(tool_calls))
        tool_results = []
        for tc in tool_calls:
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            result = handle_tool_call(tc.name, args, SKILL_MAP)
            logger.info("  %s(%s) → %d chars", tc.name, args, len(result))
            tool_results.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": result,
            })
        resp = oai.responses.create(
            model=model,
            instructions=instructions,
            input=resp.output + tool_results,
            tools=resp_tools or [],
        )

    # Extract text from response output
    assistant_text = resp.output_text or ""

    # Return the response — platform will replace IDs
    return JSONResponse(content={
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}{uuid.uuid4().hex[:14]}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant_text}],
        }],
        "usage": {
            "input_tokens": resp.usage.input_tokens if resp.usage else 0,
            "output_tokens": resp.usage.output_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        },
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8088"))
    logger.info("Starting on port %d (model=%s, source=%s)", port, MODEL, config.source)
    uvicorn.run(app, host="0.0.0.0", port=port)
