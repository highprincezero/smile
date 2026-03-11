"""
Smile-Agent API Server — standalone FastAPI backend.

Run with:  uvicorn server:app --reload --port 8000
Expose:    ngrok http 8000  (or cloudflare tunnel)
"""

import os
import json

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

import brain  # smile-agent brain

app = FastAPI(title="Smile-Agent API")

# Allow all origins so ngrok/cloudflare tunnels work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_SYSTEM_PROMPT = "You are an intelligent assistant."


@app.post("/api/chat")
async def chat(request: Request):
    """Chat via smile-agent brain — streams response via SSE."""
    body = await request.json()
    user_messages = body.get("messages", [])
    system_prompt = body.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    async def generate():
        for token in brain.chat_stream(user_messages, API_KEY, system_prompt):
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
