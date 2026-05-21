"""Extract plain text from a user-uploaded file (PDF / CSV / text) so it can be
attached to a chat as supplementary data for comparison, trend analysis, etc."""

import io

MAX_CHARS = 20000  # cap injected text so the prompt stays reasonable


def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as e:  # noqa: BLE001
            return f"[could not parse PDF: {e}]"
    else:
        # csv / tsv / txt / md / json — decode as text
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    return text[:MAX_CHARS]
