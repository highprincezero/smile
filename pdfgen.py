"""Build a simple PDF report (title + bar chart + body text) for chat exports.

Uses fpdf2 core fonts (latin-1), so non-latin glyphs (₱, emoji) are sanitized.
The chart is drawn as horizontal bars from (label, value) rows — so the exported
PDF visually includes the chart, not just text.
"""

import re

from fpdf import FPDF

_GREEN = (34, 197, 94)


def _latin1(s: str) -> str:
    if not s:
        return ""
    s = s.replace("₱", "PHP ")  # ₱
    return s.encode("latin-1", "ignore").decode("latin-1")


def parse_chart(data: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in (data or "").splitlines():
        line = line.strip()
        if not line:
            continue
        sep = "=" if "=" in line else ("," if "," in line else None)
        if not sep:
            continue
        label, _, val = line.rpartition(sep)
        m = re.search(r"-?\d+(?:\.\d+)?", val.replace(",", ""))
        if not m:
            continue
        rows.append((label.strip(), float(m.group())))
    return rows


def build_pdf(title: str, chart: list[tuple[str, float]], body_text: str) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _latin1(title or "Smile Export"))
    pdf.ln(2)

    if chart:
        bar_max = 110.0
        maxv = max(v for _, v in chart) or 1.0
        for label, v in chart:
            y = pdf.get_y()
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(48, 6, _latin1(label)[:30])
            pdf.set_fill_color(*_GREEN)
            pdf.rect(pdf.get_x(), y + 1.2, max(0.5, bar_max * v / maxv), 3.6, style="F")
            pdf.set_xy(pdf.get_x() + bar_max + 3, y)
            pdf.cell(25, 6, f"{v:g}")
            pdf.ln(7)
        pdf.ln(4)

    if body_text:
        pdf.set_font("Courier", "", 9)
        pdf.multi_cell(0, 5, _latin1(body_text))

    return bytes(pdf.output())
