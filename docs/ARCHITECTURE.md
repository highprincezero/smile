# Smile-Agent — Architecture & Blueprint

> **BLUF.** Smile is the FastAPI backend powering the Greenfy.ai chatbot. After Phase 1 (this branch), the primary chat endpoint is driven by the **Claude Agent SDK** — Claude can autonomously call Smile's NLP and bond-data capabilities as tools. The original OpenAI/OpenRouter path is preserved as a parallel fallback at `/api/chat-backdoor`. HuggingFace deterministic NLP and the bond data pipeline keep their direct REST endpoints untouched.

---

## 1. System context

| Caller | What it calls | Purpose |
|---|---|---|
| Greenfy marketing site (`highprincezero/greenfy`) — `src/components/Navbar.tsx` | `/api/chat` (SSE) | Inline "Smile" chat widget |
| Greenfy marketing site — `src/lib/analyzeBond.ts` | `/api/analyze-bond` | Direct bond data fetch (frontend-initiated; bypasses chat) |
| Cloudflare tunnel (`jade-lean-geo-desire.trycloudflare.com`) | All `/api/*` endpoints | Exposes localhost:8000 to the live Greenfy site |
| Local dev | Vite proxy in Greenfy dev server | Routes to `localhost:8000` when `VITE_SMILE_API_URL` is unset |

Smile does NOT talk to: a database, a job queue, a CDN, or any third-party storage. It is stateless except for the in-memory bond cache (`cachetools`).

---

## 2. High-level architecture (ASCII)

```
                  ┌──────────────────────────────────────────────┐
                  │              Greenfy.ai frontend             │
                  │  Navbar.tsx (chat) | analyzeBond.ts (button) │
                  └────────────────┬─────────────────────────────┘
                                   │  HTTPS (Cloudflare tunnel)
                                   ▼
                  ┌──────────────────────────────────────────────┐
                  │            FastAPI (server.py)               │
                  │                                              │
                  │  POST /api/chat            ─┐                │
                  │  POST /api/chat-backdoor   ─┤  Chat surface  │
                  │                             │                │
                  │  POST /api/analyze-bond    ─┐                │
                  │  POST /api/analyze         ─┤  Deterministic │
                  │  POST /api/qa              ─┤  REST endpoints│
                  │  POST /api/keywords        ─┤                │
                  │  POST /api/summarize       ─┘                │
                  │  GET  /api/health                            │
                  └────┬───────────────┬──────────────────┬──────┘
                       │               │                  │
                       ▼               ▼                  ▼
              ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
              │ smile_agent.py │ │   brain.py   │ │   bonds/         │
              │ (Claude path)  │ │  (legacy +   │ │ (deterministic)  │
              │                │ │   HF)        │ │                  │
              │ Claude Agent   │ │ OpenAI SDK   │ │ httpx + BS4 +    │
              │ SDK + MCP      │ │ → OpenRouter │ │ cachetools       │
              │ tools          │ │ + HF models  │ │ → PDS sources    │
              └────┬───────────┘ └──────────────┘ └──────────────────┘
                   │
                   │ tool calls (via SDK MCP)
                   ▼
              ┌──────────────────────────────────────────────────┐
              │ smile_tools (in-process SDK MCP server)          │
              │   • extract_keywords  → brain.extract_keywords   │
              │   • document_qa       → brain.document_qa        │
              │   • summarize         → brain.summarize          │
              │   • analyze_bond      → POST /api/analyze-bond   │
              │                          (self-call via httpx)   │
              └──────────────────────────────────────────────────┘
```

---

## 3. Endpoint map

| Method | Path | Provider | Status | Purpose |
|---|---|---|---|---|
| POST | `/api/chat` | **Claude Agent SDK** (`claude-opus-4-7`) | **Primary chat** | Streamed SSE; agent can call any of the 4 registered tools |
| POST | `/api/chat-backdoor` | OpenAI SDK → OpenRouter (`gpt-4o-mini`) | Fallback chat | Streamed SSE; runs `is_on_topic_openai` gate before chat |
| POST | `/api/analyze-bond` | `bonds/` (httpx + BS4 + cachetools) | Direct | Bond data fetcher (also wrapped as `analyze_bond` tool) |
| POST | `/api/analyze` | HF transformers (KeyBERT + RoBERTa + BART) | Direct | Full document analysis pipeline |
| POST | `/api/qa` | HF RoBERTa (`deepset/roberta-base-squad2`) | Direct | Document Q&A |
| POST | `/api/keywords` | HF KeyBERT | Direct | Keyword extraction |
| POST | `/api/summarize` | HF BART (`facebook/bart-large-cnn`) | Direct | Summarization |
| GET | `/api/health` | — | — | Liveness check |

---

## 4. Module map

| File / dir | Role | Loaded by |
|---|---|---|
| `server.py` | FastAPI app + endpoint definitions + CORS + bonds router include | Uvicorn entry |
| `smile_agent.py` | Claude Agent SDK path: 4 `@tool` wrappers + MCP server + `ClaudeAgentOptions` + `chat_stream_claude` | `server.py` |
| `brain.py` | OpenAI backdoor (`chat_stream_openai`, `is_on_topic_openai`) + HF deterministic helpers (`extract_keywords`, `document_qa`, `summarize`, `analyze_document`) | `server.py`, `smile_agent.py` |
| `bonds/router.py` | `/api/analyze-bond` FastAPI router | `server.py` |
| `bonds/cache.py`, `normalize.py`, `schema.py`, `sources/` | Bond pipeline internals | `bonds/router.py` |
| `prompt/base.txt` | Claude + OpenAI system prompt (topic restriction + injection defense) | `smile_agent.py`, `brain.py` |
| `prompt/gate.txt` | OpenAI gate classifier prompt | `brain.py` (backdoor only) |
| `prompt/off_topic.txt` | Canned off-topic reply | `brain.py` (backdoor only) |
| `prompt/normalize_bond.txt` | LLM normalize prompt for bond data | `bonds/normalize.py` |
| `prompt/greeting.txt` | Initial chat greeting | Frontend (loaded statically) |
| `skills/bond-analysis/SKILL.md` | Claude Agent SDK skill overlay — when/how to use bond tools | Auto-loaded by SDK (`setting_sources=["project"]`) |
| `.env` | Local secrets + model config | `python-dotenv` |

---

## 5. Request flows

### 5.1 Primary chat flow (Claude path)

```
1. POST /api/chat  {"messages":[{"role":"user","text":"..."}]}
2. server.py:chat() → smile_agent.chat_stream_claude(messages)
3. ClaudeSDKClient opens session, query(user_text)
4. Claude reasons:
     ├── If topic permitted (per base.txt rules):
     │     ├── May call tool(s) via MCP:
     │     │     • extract_keywords / document_qa / summarize
     │     │     • analyze_bond (httpx → /api/analyze-bond self-call)
     │     └── Synthesizes response
     └── If off-topic / injection attempt:
           └── Refuses per base.txt guardrails
5. SDK streams response → chat_stream_claude yields text chunks
6. server.py wraps each chunk: `data: {"token":"..."}\n\n`
7. Final `data: [DONE]\n\n`
```

### 5.2 Backdoor chat flow (OpenAI path)

```
1. POST /api/chat-backdoor  {"messages":[...]}
2. server.py:chat_backdoor() → brain.chat_stream_openai(messages, API_KEY)
3. is_on_topic_openai(...) → cheap classifier call
     ├── on_topic → proceed to chat model
     └── off_topic → yield OFF_TOPIC_RESPONSE, return
4. OpenAI streaming chat completion → yield deltas
5. server.py wraps each token: `data: {"token":"..."}\n\n`
6. Final `data: [DONE]\n\n`
```

### 5.3 Direct bond fetch (no chat agent)

```
1. POST /api/analyze-bond  {"security_id":"...","type":"government"|"corporate"}
2. bonds/router.py:analyze_bond() validates request
3. cache.get(...) — if cached and not refresh, return
4. fetch_government / fetch_corporate — scrape PDS
5. normalize(...) — LLM-normalize raw snippet to schema
6. cache.put(...) — store result
7. Return JSON
```

### 5.4 Tool-driven bond fetch (within chat)

```
User: "Analyze SM Bond Series 7 for me"
  ↓
Claude (loaded skills/bond-analysis/SKILL.md):
  → Tool call: analyze_bond(security_id="SM Bond Series 7", type="corporate")
  ↓
smile_agent.py:analyze_bond() → httpx POST localhost:8000/api/analyze-bond
  ↓
bonds/router.py:analyze_bond() — same as 5.3
  ↓
Tool result returns to Claude as text (JSON)
  ↓
Claude formats as markdown table per skill workflow
  ↓
Streamed back to user via /api/chat SSE
```

---

## 6. Tool surface (Claude Agent SDK)

All four tools live in `smile_agent.py`, decorated with `@tool`, bundled into a single SDK MCP server (`smile_tools`), and explicitly allowlisted in `ClaudeAgentOptions.allowed_tools`.

| Tool name (MCP) | Schema | Backing function | Cost |
|---|---|---|---|
| `mcp__smile__extract_keywords` | `{text: str, top_n: int}` | `brain.extract_keywords()` — KeyBERT | Local model — free per call after cold start |
| `mcp__smile__document_qa` | `{question: str, context: str}` | `brain.document_qa()` — RoBERTa squad2 | Local model — free per call after cold start |
| `mcp__smile__summarize` | `{text: str, max_length: int}` | `brain.summarize()` — BART CNN | Local model — free per call after cold start |
| `mcp__smile__analyze_bond` | `{security_id: str, type: str}` | httpx POST `/api/analyze-bond` self-call | LLM normalize call inside (OpenAI / OpenRouter) |

Tool registration pattern:

```python
smile_tools = create_sdk_mcp_server(
    name="smile_tools",
    version="1.0.0",
    tools=[extract_keywords, document_qa, summarize, analyze_bond],
)

options = ClaudeAgentOptions(
    model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
    system_prompt=(_PROMPT_DIR / "base.txt").read_text(),
    mcp_servers={"smile": smile_tools},
    allowed_tools=["mcp__smile__extract_keywords", ...],
    setting_sources=["project"],
)
```

---

## 7. Skill overlay

`skills/` at the repo root is auto-discovered because `setting_sources=["project"]`.

| Skill | Trigger description (frontmatter) | What it teaches Claude |
|---|---|---|
| `bond-analysis` | "Use when the user mentions a specific bond by security ID, ticker, or asks for yield, maturity, rating, or bond data analysis." | Call `analyze_bond` → optionally `summarize` long fields → render as markdown table with yield/maturity/rating/issuer + ₱ currency + source URL citation. Do NOT call `extract_keywords` on bond data. |

**Progressive disclosure** — only the frontmatter `description:` lives in Claude's context at idle. The body is loaded on-demand when Claude judges the description matches the user's intent.

---

## 8. Auth & secrets model

| Provider | Auth method | Where set |
|---|---|---|
| Anthropic (Claude Agent SDK) | **Claude Max local auth** (preferred) OR `ANTHROPIC_API_KEY` env var | `.env` line commented out — SDK inherits local Claude Code auth |
| OpenAI / OpenRouter (backdoor + bond normalize) | API key | `.env` `OPENAI_API_KEY` or `OPENROUTER_API_KEY` |
| Pinecone (legacy, not loaded in current flow) | API key | `.env` `PINECONE_API_KEY` (unused in Phase 1) |

`.env` is git-ignored (`.gitignore` entries: `.env`, `.env.*`).

---

## 9. Deployment topology

```
    Browser (Greenfy.ai)
         │
         ▼
    Cloudflare tunnel (cloudflared)
         │
         ▼  forwards to:
    localhost:8000 (uvicorn server:app)
         │
         ├─ smile_agent.py → Claude Agent SDK → (Claude Max account auth via local Claude Code)
         ├─ brain.py       → openai.OpenAI(base_url="openrouter.ai") + lazy HF models
         └─ bonds/         → httpx → PDS sources + normalize call → cachetools
```

| Process | Command | Notes |
|---|---|---|
| API server | `uvicorn server:app --port 8000 --reload` | dev only — production uses `--workers N` |
| Tunnel | `cloudflared tunnel --url http://localhost:8000` | gives the `jade-lean-geo-desire.trycloudflare.com` URL |
| Frontend | `npm run dev` (Greenfy repo) | hits `VITE_SMILE_API_URL` from `.env` (defaults to the tunnel URL) |

---

## 10. Modularity principles

These shaped Phase 1's scope:

| Principle | What it means | Trade-off taken |
|---|---|---|
| **Don't fork the live API surface** | Existing endpoints keep their contracts; new endpoints are additive | `/api/chat-backdoor` (additive) rather than `/api/chat-openai` (rename) |
| **Wire format stability** | Frontend doesn't change. SSE chunks remain `data: {"token":"..."}\n\n` regardless of provider | `smile_agent.chat_stream_claude` translates Anthropic stream events into the existing chunk shape |
| **Capability = function + tool wrapper** | Each capability is a `brain.py` function AND a thin `@tool` wrapper. Direct REST users keep using REST; agent users can also call it | Doubled surface but zero migration burden for the frontend |
| **Tools call their own REST endpoint** | `analyze_bond` tool does `httpx → localhost:8000/api/analyze-bond` instead of `import bonds.router` | Slight overhead, but the bond pipeline stays untouched (modularity) |
| **System prompt is the only guardrail in the Claude path** | No separate gate-LLM; rely on `prompt/base.txt`'s topic-restriction + injection-defense rules | Cheaper, simpler, and Claude 4.7 follows system prompts closely |
| **Skills are optional, not required** | The system works without `skills/`. Skills are workflow overlays for common patterns | First skill (`bond-analysis`) is illustrative; more can be added on demand |

---

## 11. Roadmap

| Phase | Scope | Status | Branch |
|---|---|---|---|
| **0** | Baseline: OpenAI-only chat at `/api/chat`, bonds pipeline as standalone REST | Pre-existing | `main` (merged via PR #4) |
| **1** | Claude Agent SDK chat path + 4 capability tools + bond-analysis skill | **DONE** (this doc) | `sdk-conversion-opus` (commit `dc11524`) |
| **2 (optional)** | Deprecate `/api/chat-backdoor`; frontend always uses Claude. Frontend `analyzeBond.ts` deleted (Claude calls bond tool instead of frontend) | Pending decision | TBD |
| **3 (optional)** | Multi-agent: sub-agent for bond-research, sub-agent for ESG-explainer, coordinator on `/api/chat`. Add memory store for user preferences. | Speculative | TBD |
| **4 (optional)** | Move from Cloudflare tunnel → managed deployment (Fly.io / Railway / Render). Add observability (Sentry). | Speculative | TBD |

---

## 12. File structure (post-Phase 1)

```
smile/
├── server.py                    # FastAPI app + routes
├── smile_agent.py               # NEW — Claude Agent SDK path
├── brain.py                     # OpenAI backdoor + HF deterministic helpers
├── bonds/
│   ├── router.py                # /api/analyze-bond
│   ├── cache.py
│   ├── normalize.py
│   ├── schema.py
│   └── sources/                 # PDS scrapers
├── prompt/
│   ├── base.txt                 # System prompt (both paths)
│   ├── gate.txt                 # Backdoor-only classifier
│   ├── off_topic.txt            # Backdoor-only canned reply
│   ├── greeting.txt             # Frontend greeting copy
│   └── normalize_bond.txt       # Bond normalize prompt
├── skills/                      # NEW — Agent SDK skill overlays
│   └── bond-analysis/
│       └── SKILL.md
├── docs/
│   └── ARCHITECTURE.md          # This file
├── CLAUDE.md                    # Project instructions (refreshed in Phase 1)
├── requirements.txt             # Adds claude-agent-sdk
├── .env                         # ANTHROPIC_MODEL + SMILE_SELF_BASE added; KEY blank (Max auth)
├── .gitignore                   # Adds BUP-*/, .venv, .claude/
├── smile.py                     # Legacy Streamlit app (not used by API)
├── smile_old.py                 # Legacy
├── index.html                   # Static landing
└── README.md
```

---

## 13. Smoke test

```bash
# In the worktree:
cd ~/Developer/smile/.claude/worktrees/sdk-conversion-opus

# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn server:app --port 8000 --reload &

# Smoke tests
curl -N -X POST http://localhost:8000/api/health
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","text":"What is a green bond?"}]}'

# Off-topic — should refuse per base.txt rules
curl -N -X POST http://localhost:8000/api/chat \
  -d '{"messages":[{"role":"user","text":"Write me a poem about cats"}]}'

# Backdoor regression — same body, different endpoint
curl -N -X POST http://localhost:8000/api/chat-backdoor \
  -d '{"messages":[{"role":"user","text":"What is a green bond?"}]}'
```

Expected:
- `/api/health` → `{"status":"ok","engine":"smile-agent"}`
- `/api/chat` → SSE stream of `data: {"token":"..."}\n\n` chunks ending with `data: [DONE]\n\n`
- Off-topic → canned-style refusal (per `base.txt` guardrails)
- `/api/chat-backdoor` → same shape, different model

---

## 14. References

- [Claude Agent SDK Python docs](https://docs.claude.com/en/api/agent-sdk-python)
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/)
- [FastAPI Streaming Responses](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- Smile session memory: `~/.claude/projects/-Users-red/memory/session_greenfy_smile_20260515.md`
- Drafts snapshot: `~/.claude/projects/-Users-red/memory/session_greenfy_smile_20260515_drafts.md`
