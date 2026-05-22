"""
Smile-Agent API Server — standalone FastAPI backend.

Run with:  uvicorn server:app --reload --port 8000
Expose:    ngrok http 8000  (or cloudflare tunnel)
"""

import os
import json
import secrets
from collections import OrderedDict

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

import brain  # smile-agent brain
import fileio
import pdfgen
from bonds.router import router as bonds_router
from smile_agent import chat_stream_claude

app = FastAPI(title="Smile-Agent API")
app.include_router(bonds_router)

# Allow all origins so ngrok/cloudflare tunnels work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY")


# In-memory export store (last 100), for download links the chat hands the user.
_EXPORTS: "OrderedDict[str, dict]" = OrderedDict()
_EXPORT_MIME = {"csv": "text/csv", "md": "text/markdown", "txt": "text/plain", "json": "application/json"}


@app.post("/api/export")
async def make_export(request: Request):
    """Store content for download; return a relative path the chat embeds as a link.
    format=pdf renders a report (title + bar chart from `chart` + `content` text)."""
    body = await request.json()
    fmt = (body.get("format") or "txt").lower()
    eid = secrets.token_urlsafe(8)
    filename = body.get("filename") or f"export.{fmt}"

    if fmt == "pdf":
        content = pdfgen.build_pdf(
            title=body.get("title") or filename,
            chart=pdfgen.parse_chart(body.get("chart") or ""),
            body_text=(body.get("content") or "")[:200_000],
        )
        mime = "application/pdf"
    else:
        content = (body.get("content") or "")[:500_000].encode("utf-8")
        mime = _EXPORT_MIME.get(fmt, "text/plain")

    _EXPORTS[eid] = {"filename": filename, "content": content, "mime": mime}
    while len(_EXPORTS) > 100:
        _EXPORTS.popitem(last=False)
    return {"path": f"/api/download/{eid}", "filename": filename}


@app.get("/api/download/{export_id}")
async def download(export_id: str):
    item = _EXPORTS.get(export_id)
    if not item:
        return JSONResponse(status_code=404, content={"error": "not found or expired"})
    return Response(
        content=item["content"],
        media_type=item["mime"],
        headers={"Content-Disposition": f'attachment; filename="{item["filename"]}"'},
    )


@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)):
    """Parse an uploaded file (PDF / CSV / text) to plain text the chat can attach."""
    raw = await file.read()
    text = fileio.extract_text(file.filename or "upload", raw)
    return {"name": file.filename or "upload", "text": text, "chars": len(text)}


@app.post("/api/chat")
async def chat(request: Request):
    """Chat via Claude Agent SDK (primary path) — streams response via SSE.
    Optional `attachment` ({name, text}) supplies a user-uploaded file as context."""
    body = await request.json()
    messages = body.get("messages", [])
    attachment = body.get("attachment")

    async def generate():
        async for event in chat_stream_claude(messages, attachment=attachment):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat-backdoor")
async def chat_backdoor(request: Request):
    """Legacy OpenAI/OpenRouter path — kept as fallback."""
    body = await request.json()
    user_messages = body.get("messages", [])

    async def generate():
        for token in brain.chat_stream_openai(user_messages, API_KEY):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """Smile-agent full document analysis: keywords + Q&A + summary."""
    content = (await file.read()).decode("utf-8", errors="ignore")
    result = brain.analyze_document(content)
    return result


@app.post("/api/summarize")
async def summarize_document(file: UploadFile = File(...)):
    """Smile-agent summarization via BART."""
    content = (await file.read()).decode("utf-8", errors="ignore")
    summary = brain.summarize(content)
    return {"summary": summary}


@app.post("/api/qa")
async def document_qa(request: Request):
    """Smile-agent domain Q&A — ask a question about provided context."""
    body = await request.json()
    question = body.get("question", "")
    context = body.get("context", "")

    if not question or not context:
        return {"error": "Both 'question' and 'context' are required."}

    result = brain.document_qa(question, context)
    return {
        "answer": result["answer"],
        "confidence": round(result["score"], 3),
    }


@app.post("/api/keywords")
async def extract_keywords(file: UploadFile = File(...)):
    """Smile-agent keyword extraction via KeyBERT."""
    content = (await file.read()).decode("utf-8", errors="ignore")
    keywords = brain.extract_keywords(content)
    return {"keywords": [{"keyword": kw, "score": round(s, 3)} for kw, s in keywords]}


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "smile-agent"}
