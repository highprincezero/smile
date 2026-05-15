# Smile-Agent — Project Instructions

## Overview
Smile is a FastAPI backend powering the Greenfy.ai chatbot. It serves a Claude-driven chat agent with NLP helpers (KeyBERT, RoBERTa QA, BART) and a Philippine bond data pipeline.

## Architecture
| Path | Provider | Purpose |
|---|---|---|
| `/api/chat` | **Claude Agent SDK** (`claude-opus-4-7`) | Primary chat endpoint. All Smile capabilities registered as agent tools. |
| `/api/chat-backdoor` | OpenAI → OpenRouter (`gpt-4o-mini`) | Fallback chat path. |
| `/api/analyze-bond` | httpx + BS4 + cachetools | Bond data fetcher (also called by the Claude agent as a tool). |
| `/api/analyze`, `/api/qa`, `/api/keywords`, `/api/summarize` | HuggingFace transformers | Deterministic NLP — direct REST. |
| `/api/health` | — | Health check. |

## Project structure
- `smile_agent.py` — Claude Agent SDK path (tools, MCP server, chat_stream_claude)
- `brain.py` — OpenAI backdoor path + HF deterministic helpers
- `bonds/` — bond data pipeline (router, cache, sources, normalize)
- `prompt/` — system + classifier prompts (`base.txt`, `gate.txt`, `off_topic.txt`, ...)
- `skills/` — Agent SDK skill overlays (markdown workflows)
- `server.py` — FastAPI wiring

## Running locally
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

Required env: `ANTHROPIC_API_KEY` (Claude path) + `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (backdoor + bond normalize).

## Adding new agent capabilities
1. Add the function to `brain.py` (or wherever it belongs).
2. Wrap it as a `@tool` in `smile_agent.py` and append to the `create_sdk_mcp_server(tools=[...])` list.
3. Add the tool name to `allowed_tools` in `ClaudeAgentOptions`.
4. (Optional) Drop a `skills/<name>/SKILL.md` workflow overlay.
