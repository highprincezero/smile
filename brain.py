"""
Smile-Agent Brain — extracted from smile.py for API consumption.

This is the core intelligence layer that server.py calls.
All AI capabilities route through here.
"""

import openai
import os

# KeyBERT + transformers are imported lazily inside getter functions
# so the server starts fast and only loads models when first needed.

# ---------- lazy singletons ----------
_kb = None
_qa = None
_summarizer = None


def _get_keybert():
    global _kb
    if _kb is None:
        from keybert import KeyBERT
        _kb = KeyBERT()
    return _kb


def _get_qa():
    global _qa
    if _qa is None:
        from transformers import pipeline as hf_pipeline
        _qa = hf_pipeline("question-answering", model="deepset/roberta-base-squad2")
    return _qa


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline as hf_pipeline
        _summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


# ---------- chat (from smile.py Global mode) ----------

def chat(messages: list[dict], api_key: str, system_prompt: str = None) -> str:
    """
    Smile-agent chat — mirrors smile.py's Global chat.
    Uses openai.ChatCompletion pattern from the original agent.
    """
    client = openai.OpenAI(api_key=api_key)

    openai_messages = [
        {"role": "system", "content": system_prompt or "You are an intelligent assistant."}
    ]
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "assistant"
        openai_messages.append({"role": role, "content": msg.get("text", "")})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
    )
    return response.choices[0].message.content


def chat_stream(messages: list[dict], api_key: str, system_prompt: str = None):
    """
    Streaming version of smile-agent chat. Yields token strings.
    """
    client = openai.OpenAI(api_key=api_key)

    openai_messages = [
        {"role": "system", "content": system_prompt or "You are an intelligent assistant."}
    ]
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "assistant"
        openai_messages.append({"role": role, "content": msg.get("text", "")})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ---------- keyword extraction (from smile.py analyze mode) ----------

def extract_keywords(text: str, top_n: int = 8) -> list[tuple[str, float]]:
    """Smile-agent KeyBERT keyword extraction."""
    kb = _get_keybert()
    return kb.extract_keywords(text, top_n=top_n)


# ---------- document Q&A (from smile.py Domain mode) ----------

def document_qa(question: str, context: str) -> dict:
    """
    Smile-agent domain Q&A — deepset/roberta-base-squad2.
    Returns {"answer": str, "score": float}
    """
    qa = _get_qa()
    return qa({"question": question, "context": context})


# ---------- summarization (from smile.py analyze mode) ----------

def summarize(text: str, max_length: int = 150) -> str:
    """Smile-agent summarization — facebook/bart-large-cnn."""
    summarizer = _get_summarizer()
    truncated = text[:3000]  # BART token limit
    result = summarizer(truncated, max_length=max_length, min_length=30, do_sample=False)
    return result[0]["summary_text"]


# ---------- full document analysis (from smile.py analyze mode) ----------

def analyze_document(text: str, top_n: int = 8) -> dict:
    """
    Smile-agent full analysis: keywords + Q&A for each keyword + summary.
    This is the complete pipeline from smile.py's 'analyze' flow.
    """
    keywords = extract_keywords(text, top_n=top_n)

    insights = []
    for kw, score in keywords:
        answer = document_qa(f"What is {kw}?", text)
        insights.append({
            "keyword": kw,
            "relevance": round(score, 3),
            "answer": answer["answer"],
            "confidence": round(answer["score"], 3),
        })

    summary = summarize(text)

    return {
        "keywords": insights,
        "summary": summary,
    }
