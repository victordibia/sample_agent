"""Optimization-ready hosted agent using the Responses protocol SDK.

Uses azure-ai-agentserver-responses for proper protocol handling (fixes portal
display issues) while supporting agent optimization via load_config().

Normal operation (no optimization):
    Uses the defaults you specify below — behaves like a regular chat agent.

During optimization:
    The optimization service injects OPTIMIZATION_CONFIG / AGENT_OPTIMIZATION_CONFIG.
    load_config() picks them up and returns the candidate's config.
"""

import asyncio
import logging
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)
from azure.ai.agentserver.responses.models import (
    MessageContentInputTextContent,
    MessageContentOutputTextContent,
)

from agent_optimization import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimize-agent")

# ── Config ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant.

Be concise, accurate, and helpful."""

config = load_config(
    default_instructions=SYSTEM_PROMPT,
    default_model=os.getenv("MODEL_DEPLOYMENT_NAME")
    or os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    or "gpt-4.1-mini",
)

MODEL = config.model or os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
ALL_INSTRUCTIONS = config.compose_instructions()

logger.info(
    "Config loaded (source=%s, model=%s)", config.source, MODEL
)

# ── Foundry client ────────────────────────────────────────────────────────────

_endpoint = (
    os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    or ""
)
if not _endpoint:
    raise EnvironmentError(
        "FOUNDRY_PROJECT_ENDPOINT or AZURE_AI_PROJECT_ENDPOINT must be set."
    )

_credential = DefaultAzureCredential()
_project_client = AIProjectClient(endpoint=_endpoint, credential=_credential)
_responses_client = _project_client.get_openai_client().responses

# ── Agent server ──────────────────────────────────────────────────────────────

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20),
)


def _build_input(current_input: str, history: list) -> list[dict]:
    """Build Responses API input from conversation history and current message."""
    input_items = []
    for item in history:
        if hasattr(item, "content") and item.content:
            for content in item.content:
                if isinstance(content, MessageContentOutputTextContent) and content.text:
                    input_items.append({"role": "assistant", "content": content.text})
                elif isinstance(content, MessageContentInputTextContent) and content.text:
                    input_items.append({"role": "user", "content": content.text})
    input_items.append({"role": "user", "content": current_input})
    return input_items


@app.response_handler
async def handler(
    request: CreateResponse,
    context: ResponseContext,
    _cancellation_signal: asyncio.Event,
):
    """Forward user input to the model with optimized instructions."""
    user_input = await context.get_input_text() or "Hello!"
    history = await context.get_history()

    logger.info("Processing request %s (model=%s, source=%s)",
                context.response_id, MODEL, config.source)

    input_items = _build_input(user_input, history)

    response = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _responses_client.create(
            model=MODEL,
            instructions=ALL_INSTRUCTIONS,
            input=input_items,
            store=False,
        ),
    )

    return TextResponse(context, request, text=response.output_text)


app.run()
