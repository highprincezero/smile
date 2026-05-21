# Smile ↔ Greenfy — Integration & Operations

> **BLUF.** Smile is the FastAPI backend; Greenfy.ai is the frontend. Greenfy's chat widget and dashboard call Smile's `/api/chat`, which is now driven by the **Claude Agent SDK**. The two are bridged over a **Cloudflare quick tunnel** to a Mac mini (`gZeroMacMiniM4`) running uvicorn on `localhost:8000`. This doc captures the full system, both directions of the integration, the infra, and the open bond-data work — as of **2026-05-21**.

---

## 1. Repos & ownership

| Repo | Path | Origin | Manager | Branch |
|---|---|---|---|---|
| Smile (backend) | `~/Developer/smile` | `highprincezero/smile` | Hand-authored | `main` (= ex `sdk-conversion-opus`, merged) |
| Greenfy (frontend) | `~/Desktop/Greenfy/repo/greenfy` | `highprincezero/greenfy` | **Lovable** (gpt-engineer-app bot) | `main` |

> GitHub: two local accounts. `geofmhyke` is default-active but has **no** access to these repos — must `gh auth switch --user highprincezero` to push/pull, then switch back.

---

## 2. System diagram

```
   ┌─────────────────────────────────────────────┐
   │            Greenfy.ai frontend               │
   │  Navbar.tsx (chat widget)                    │
   │  DashboardSmile.tsx (dashboard chat)         │
   │  analyzeBond.ts (direct bond fetch, if used) │
   └───────────────┬──────────────────────────────┘
                   │  HTTPS / SSE
                   │  SMILE_API = VITE_SMILE_API_URL
                   ▼
   ┌─────────────────────────────────────────────┐
   │  Cloudflare quick tunnel (cloudflared)       │
   │  identifier-extend-album-joseph              │
   │       .trycloudflare.com                     │
   └───────────────┬──────────────────────────────┘
                   │  → forwards to
                   ▼
   ┌─────────────────────────────────────────────┐
   │  gZeroMacMiniM4 : localhost:8000             │
   │  uvicorn server:app --reload                 │
   │                                              │
   │  /api/chat            → smile_agent (Claude) │
   │  /api/chat-backdoor   → brain (OpenAI)       │
   │  /api/analyze-bond    → bonds/ (PDS)         │
   │  /api/analyze|qa|keywords|summarize → HF     │
   │  /api/health                                 │
   └───────────────┬──────────────────────────────┘
                   │ self-call (httpx, SMILE_SELF_BASE)
                   ▼
   ┌─────────────────────────────────────────────┐
   │  analyze_bond tool → /api/analyze-bond       │
   │     → PDS data source (see §7)               │
   └─────────────────────────────────────────────┘
```

Also reachable directly over **Tailscale**: `http://gzeromacminim4:8000` and `ssh red@gzeromacminim4` (no public tunnel needed for internal use).

---

## 3. Integration — both directions

### 3.1 Greenfy → Smile (chat)
| Item | Value |
|---|---|
| Callers | `src/components/Navbar.tsx`, `src/pages/DashboardSmile.tsx` |
| Endpoint | `POST ${SMILE_API}/api/chat` |
| Request | `{ "messages": [ { "role": "user"|"bot", "text": "…" } ] }` |
| Response | SSE stream: `data: {"token":"…"}\n\n` … terminated by `data: [DONE]\n\n` |
| Backend | `smile_agent.chat_stream_claude()` → Claude Agent SDK (`claude-opus-4-7`) |

### 3.2 Greenfy → Smile (direct bond fetch)
| Item | Value |
|---|---|
| Caller | `src/lib/analyzeBond.ts` (frontend-initiated; bypasses chat) |
| Endpoint | `POST ${SMILE_API}/api/analyze-bond` |
| Request | `{ "security_id": "…", "type": "government"|"corporate" }` |
| Response | `BondAnalysis` JSON, or `{ "error": { "code", "message" } }` |

### 3.3 Smile → Smile ("vice-versa" / self-call)
| Item | Value |
|---|---|
| Trigger | Claude calls the `analyze_bond` MCP tool during a chat |
| Mechanism | Tool does `httpx POST ${SMILE_SELF_BASE}/api/analyze-bond` (default `http://localhost:8000`) |
| Why | Keeps the bond pipeline behind one HTTP contract; the agent reuses the same endpoint the frontend uses |

### 3.4 Wire-format contract (must stay stable)
The frontend parses `data: {"token":"…"}` lines and stops on `data: [DONE]`. **Any** chat provider (Claude or OpenAI backdoor) must emit this exact shape. `chat_stream_claude` translates Anthropic stream events into these chunks.

---

## 4. The SDK conversion (what shipped 2026-05-16 → 05-21)

| Commit | Change |
|---|---|
| `dc11524` | Added `smile_agent.py` (Claude Agent SDK path); `/api/chat` → Claude; OpenAI moved to `/api/chat-backdoor` |
| `4904a44` | `docs/ARCHITECTURE.md` blueprint |
| `c8fb6a8` | **Fix**: empty `messages:[]` crashed the SSE stream (IndexError, no `[DONE]`) |
| `5f99ded` | `start-claude.command` launcher |
| (merge) | `sdk-conversion-opus` fast-forwarded → `main`, pushed (smile origin pushed **21:48 +08**) |

### Tool surface (Claude Agent SDK / MCP)
| MCP tool | Backing | Status |
|---|---|---|
| `mcp__smile__extract_keywords` | `brain.extract_keywords` (KeyBERT) | OK |
| `mcp__smile__document_qa` | `brain.document_qa` (RoBERTa) | ⚠️ see §8 (transformers v5) |
| `mcp__smile__summarize` | `brain.summarize` (BART) | OK |
| `mcp__smile__analyze_bond` | self-call `/api/analyze-bond` | ⚠️ data source dead — honest-fix applied (§7) |

Skill `skills/bond-data-collector/SKILL.md` auto-loads via `setting_sources=["project"]`.

---

## 5. Infrastructure

### 5.1 Cloudflare tunnel (the bridge)
| Item | Value |
|---|---|
| Type | **Quick / ephemeral** tunnel (`cloudflared tunnel --url http://localhost:8000`) |
| Current URL | `https://identifier-extend-album-joseph.trycloudflare.com` |
| Previous (dead) | `jade-lean-geo-desire.trycloudflare.com` |
| ⚠️ Caveat | URL **changes on every restart / Mac sleep** — frontend must be re-pointed each time |
| Permanent fix | Named Cloudflare tunnel (stable hostname, one-time `cloudflared tunnel login`) — **not yet set up** |

### 5.2 Tailscale (internal access)
| Item | Value |
|---|---|
| Daemon | `tailscaled` as **root** LaunchDaemon (`sudo brew services start tailscale`) |
| SSH | `sudo tailscale up --ssh` |
| Reach | `ssh red@gzeromacminim4`; API at `http://gzeromacminim4:8000` |

### 5.3 URL propagation to Greenfy (when the tunnel URL changes)
| Location | Update |
|---|---|
| `Navbar.tsx` / `DashboardSmile.tsx` fallback default | committed `d7d59c8` (greenfy pushed **22:02 +08**) |
| `.env` `VITE_SMILE_API_URL` (gitignored) | updated locally |
| **Lovable env + redeploy** | ⚠️ **required for public site** — set `VITE_SMILE_API_URL` to the current tunnel and redeploy (cannot be done from the repo) |

---

## 6. Run / operate

```bash
# Backend (Mac mini, in the worktree)
cd ~/Developer/smile/.claude/worktrees/sdk-conversion-opus
nohup .venv/bin/uvicorn server:app --port 8000 --reload > /tmp/uvicorn.log 2>&1 &

# Tunnel
cloudflared tunnel --url http://localhost:8000   # prints the public URL

# Health / smoke
curl -s http://localhost:8000/api/health
curl -s -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","text":"What is a green bond?"}]}'
```

> `--reload` picks up `.py` edits but **not** pip installs — restart the process after dependency changes.

---

## 7. Bond data source — investigation & fix path

### 7.1 Why `/api/analyze-bond` returns nothing
| Layer | Finding |
|---|---|
| Endpoint / tool wiring | ✅ Correct |
| PDS page fetch | ✅ HTTP 200 |
| PDS HTML | ❌ `0` tables / `0` `<tr>` — data is **JS-rendered** (Vue + Blazor) |
| Bot protection | **Imperva Incapsula** WAF (`_Incapsula_Resource`) |
| Net effect | The BeautifulSoup scraper sees no rows → **always 404 `not_found`** for every security |

### 7.2 Honest-agent fix (applied, uncommitted)
| File | Change |
|---|---|
| `smile_agent.py` | `analyze_bond` tool: on non-200/error returns `LIVE_BOND_DATA_UNAVAILABLE` instead of a misleading 404 |
| `skills/bond-data-collector/SKILL.md` | Agent told to say "lookup offline", never estimate figures, no data table |
| Verified | Chat for `FXTN 10-65` → "data source temporarily offline… I won't guess at numbers" + general context, zero invented numbers |

### 7.3 Real fix path (downloadable PDFs on public S3 — bypasses the WAF)
Source page: `https://www.pds.com.ph/downloadable-reports/` → files on `pdswordpressbucket.s3.ap-southeast-1.amazonaws.com`.

| Report | Cadence | Fields | Use |
|---|---|---|---|
| `Corporate-Board-Summary-Price-as-of-<date>` | Month-end (latest **Apr 30 2026**) | Local ID, ISIN, CPN, YRS, Maturity, Last/Bid/Offer/Close Price, Vol | Corporate price |
| `Corporate-Board-Summary-Yield-as-of-<date>` | Month-end | yields | Corporate yield |
| `PDEx-Trade-Summary-as-of-<date>` | **Daily** (latest **May 21 2026**) | per-ticker **volume only** | Volume/activity |
| `GS-Volume-Turnover` | Monthly | government **volume only** | — |

Proposed implementation (replaces HTML scraper in `bonds/sources/`):
1. Scrape `downloadable-reports` → latest Price + Yield PDF URLs.
2. Download from S3 (plain `httpx`, no WAF).
3. `pdfplumber` parse → match row by Local ID / ISIN.
4. Map → `BondAnalysis`, cache.

| Caveat | Note |
|---|---|
| Cadence | Board Summary is **month-end**, not intraday |
| Coverage | Corporate confirmed; **government (FXTN/RTB) price/yield not in the downloadable set** (only volume) |
| Rating | **Not in PDS** at all — needs a separate source |
| New dep | `pdfplumber` (installed into the venv; not yet in `requirements.txt`) |

---

## 8. Known issues / pending

| # | Item | Severity | Status |
|---|---|---|---|
| 1 | `/api/analyze` & `/api/qa` 500 — `transformers` v5 dropped the `question-answering` pipeline | High | Pending — pin `transformers<5`, reinstall, restart |
| 2 | Chat replies **glue pre-tool narration** to the answer ("…tool first.I tried…") — `chat_stream_claude` yields text from every assistant msg incl. the one with the `ToolUseBlock` | Med | Pending — skip messages containing a `ToolUseBlock` |
| 3 | Over-explained replies (repeated bullet menus) | Med | Pending — tighten `prompt/base.txt` |
| 4 | Bond data source dead | High | Honest-fix applied (§7.2); real PDF fix scoped (§7.3) |
| 5 | Quick tunnel URL is ephemeral | Med | Pending — named tunnel |
| 6 | Public greenfy.ai still on old URL | High | Pending — Lovable env + redeploy |
| 7 | API invocation logger/dashboard | Low | Requested, not built (pure-ASGI middleware + `/dashboard`) |

---

## 9. File map (Smile)

| File | Role |
|---|---|
| `server.py` | FastAPI app + routes + CORS |
| `smile_agent.py` | Claude Agent SDK path: 4 `@tool`s, MCP server, `chat_stream_claude` |
| `brain.py` | OpenAI backdoor + HF helpers (KeyBERT/RoBERTa/BART) |
| `bonds/router.py` | `/api/analyze-bond` |
| `bonds/sources/pds_*.py` | PDS adapters (HTML scraper — **broken**, see §7) |
| `bonds/{cache,normalize,schema}.py` | Bond pipeline internals |
| `prompt/base.txt` | System prompt (chat) |
| `skills/bond-data-collector/SKILL.md` | Bond workflow skill |
| `docs/ARCHITECTURE.md` | SDK conversion blueprint |
| `docs/SMILE_GREENFY_INTEGRATION.md` | This file |
