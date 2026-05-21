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
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{_SELF_BASE}/api/analyze-bond",
                json={"security_id": args["security_id"], "type": args["type"]},
            )
        status, body = resp.status_code, resp.text
    except Exception as e:
        status, body = None, str(e)

    if status == 200:
        return {"content": [{"type": "text", "text": body}]}

    if status == 404:
        msg = (
            "BOND_NOT_FOUND: no security matched that ID in the PDS Corporate Board "
            "Summary. Do NOT guess or invent an ID. Call the list_bonds tool (pass the "
            "issuer name as query if known) to get the REAL available Local IDs, show "
            "them to the user in a table, and ask them to pick one."
        )
        return {"content": [{"type": "text", "text": msg}]}

    msg = (
        "LIVE_BOND_DATA_UNAVAILABLE: the bond data source cannot be reached right now. "
        "Tell the user live bond data lookup is temporarily offline. Do NOT invent or "
        "estimate any figures; you may still explain the bond type in general terms."
    )
    return {"content": [{"type": "text", "text": msg}]}


@tool(
    "list_bonds",
    "List available corporate bonds (with their REAL PDS Local IDs) optionally filtered "
    "by issuer or keyword. ALWAYS call this when the user asks generally about bonds, "
    "asks 'what can I look at', or names an issuer (e.g. 'Ayala') — so you present real "
    "security IDs instead of guessing.",
    {"query": str},
)
async def list_bonds(args: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(
                f"{_SELF_BASE}/api/list-bonds",
                params={"type": "corporate", "query": args.get("query", "")},
            )
        return {"content": [{"type": "text", "text": resp.text}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"LIVE_BOND_DATA_UNAVAILABLE: {e}"}]}


@tool(
    "bond_stats",
    "Compute REAL statistical analysis over the corporate bond market: summary stats "
    "(mean/median/stdev/variance/min/max for coupon, tenor, yield, price), Pearson "
    "correlations (coupon~tenor, coupon~yield, tenor~yield), linear trend / yield-curve "
    "slope, and per-issuer aggregates. Pass an issuer/keyword to scope it, or omit for "
    "the whole market. Use this whenever the user asks for correlation, variance, "
    "dispersion, trend, the yield curve, or any quantitative comparison of bonds.",
    {"issuer": str},
)
async def bond_stats(args: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=40.0) as http:
            resp = await http.get(
                f"{_SELF_BASE}/api/bond-stats",
                params={"issuer": args.get("issuer", "")},
            )
        return {"content": [{"type": "text", "text": resp.text}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"LIVE_BOND_DATA_UNAVAILABLE: {e}"}]}


# ───────── Bundle as SDK MCP server ─────────

smile_tools = create_sdk_mcp_server(
    name="smile_tools",
    version="1.0.0",
    tools=[extract_keywords, document_qa, summarize, analyze_bond, list_bonds, bond_stats],
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
        "mcp__smile__list_bonds",
        "mcp__smile__bond_stats",
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
    # Suppress pre-tool narration ("let me load the tool…"): buffer any text that
    # arrives before a tool call; once a tool runs, discard that buffer and stream
    # only the real post-tool answer. If no tool is ever used, the buffered text IS
    # the answer, so flush it at the end.
    pre_tool: list[str] = []
    tool_used = False
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_text)
        async for msg in client.receive_response():
            blocks = getattr(msg, "content", []) or []
            if any(type(b).__name__ == "ToolUseBlock" for b in blocks):
                tool_used = True
                pre_tool.clear()
                continue
            for block in blocks:
                text = getattr(block, "text", None)
                if not text:
                    continue
                if tool_used:
                    yield text
                else:
                    pre_tool.append(text)
    if not tool_used and pre_tool:
        yield "".join(pre_tool)
