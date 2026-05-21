"""
Smile-Agent — Claude Agent SDK path (Phase 1).

Wires Smile's existing capabilities as @tool functions the Claude agent can call,
and exposes chat_stream_claude() as a drop-in replacement for brain.chat_stream.

The existing brain.py (OpenAI/HF) stays alive as the backdoor path; this module
runs alongside it, not on top of it.
"""

import json
import os
from pathlib import Path

import httpx
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

import brain  # existing HF helpers

_PROMPT_DIR = Path(__file__).parent / "prompt"
_SELF_BASE = os.getenv("SMILE_SELF_BASE", "http://localhost:8000")


# ───────── Smile capabilities as Agent SDK tools ─────────

@tool(
    "extract_keywords",
    "Extract top-N keywords from a document using KeyBERT. "
    "Use when the user asks to identify key topics in a long passage.",
    {"text": str, "top_n": int},
)
async def extract_keywords(args: dict) -> dict:
    kws = brain.extract_keywords(args["text"], top_n=args.get("top_n", 8))
    return {"content": [{"type": "text", "text": json.dumps(kws)}]}


@tool(
    "document_qa",
    "Answer a question about a provided document using RoBERTa. "
    "Use when the user supplies context and asks a specific question about it.",
    {"question": str, "context": str},
)
async def document_qa(args: dict) -> dict:
    result = brain.document_qa(args["question"], args["context"])
    return {"content": [{"type": "text", "text": json.dumps(result)}]}


@tool(
    "summarize",
    "Summarize a long passage using BART. Use for documents over ~300 words.",
    {"text": str, "max_length": int},
)
async def summarize(args: dict) -> dict:
    summary = brain.summarize(args["text"], max_length=args.get("max_length", 150))
    return {"content": [{"type": "text", "text": summary}]}


@tool(
    "analyze_bond",
    "Fetch bond data by security ID. Use when the user mentions a specific bond "
    "by ID or asks for yield / maturity / rating analysis.",
    {"security_id": str, "type": str},
)
async def analyze_bond(args: dict) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            f"{_SELF_BASE}/api/analyze-bond",
            json={"security_id": args["security_id"], "type": args["type"]},
        )
    return {"content": [{"type": "text", "text": resp.text}]}


# ───────── Bundle as SDK MCP server ─────────

smile_tools = create_sdk_mcp_server(
    name="smile_tools",
    version="1.0.0",
    tools=[extract_keywords, document_qa, summarize, analyze_bond],
)


# ───────── Agent configuration ─────────

options = ClaudeAgentOptions(
    model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
    system_prompt=(_PROMPT_DIR / "base.txt").read_text(),
    mcp_servers={"smile": smile_tools},
    allowed_tools=[
        "mcp__smile__extract_keywords",
        "mcp__smile__document_qa",
        "mcp__smile__summarize",
        "mcp__smile__analyze_bond",
    ],
    setting_sources=["project"],  # auto-discovers ./skills/
)


# ───────── Streaming entry point ─────────

async def chat_stream_claude(messages: list[dict]):
    """Yield text chunks. server.py wraps each as `data: {"token": "..."}\\n\\n`."""
    if not messages:
        return
    user_text = messages[-1].get("text", "")
    if not user_text:
        return
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_text)
        async for msg in client.receive_response():
            for block in getattr(msg, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    yield text
