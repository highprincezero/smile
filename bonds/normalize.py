"""
LLM normalization step.

Takes a focused HTML/text snippet from a PDS source adapter and asks the
chat model (reusing brain.py's OpenAI/OpenRouter client) to emit a strict
JSON object conforming to BondAnalysis.

The model is forced into JSON object mode and we validate with Pydantic.
On validation failure we retry up to MAX_RETRIES times with the validator
error fed back to the model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

import brain  # reuse the OpenAI/OpenRouter client + model selection
from .schema import BondAnalysis, SecurityType

MAX_RETRIES = 2

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt" / "normalize_bond.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()


class NormalizationError(Exception):
    """LLM normalization failed after retries."""


async def normalize(
    *,
    security_id: str,
    type_: SecurityType,
    snippet: str,
    source_url: str,
    api_key: str | None = None,
) -> BondAnalysis:
    client = brain._client(api_key)
    model = brain._MODEL

    user_payload = json.dumps(
        {
            "security_id": security_id,
            "type": type_,
            "source_url": source_url,
            "snippet": snippet,
        },
        ensure_ascii=False,
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    last_error: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        if last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed schema validation: {last_error}. "
                        "Re-emit a valid JSON object. JSON only, no commentary."
                    ),
                }
            )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON ({e})"
            messages.append({"role": "assistant", "content": raw})
            continue

        # Inject server-controlled metadata before validation.
        data["source_url"] = source_url
        data["fetched_at"] = datetime.now(timezone.utc).isoformat()
        data["normalized_by"] = model
        data["raw_excerpt"] = snippet[:600]
        data.setdefault("security_id", security_id)
        data.setdefault("type", type_)

        try:
            return BondAnalysis.model_validate(data)
        except ValidationError as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": raw})
            continue

    raise NormalizationError(f"failed after {MAX_RETRIES + 1} attempts: {last_error}")
