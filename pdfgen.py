"""Build a simple PDF report (title + bar chart + body) for chat exports.

Renders markdown tables in the body as real PDF tables (not raw `|` pipes), and
the `chart` rows as horizontal bars. fpdf2 core fonts are latin-1, so non-latin
glyphs (₱, emoji) are sanitized.
"""

import re

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_GREEN = (34, 197, 94)


def _latin1(s: str) -> str:
    if not s:
        return ""
    s = s.replace("₱", "PHP ")
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


def _is_table_line(line: str) -> bool:
    # Any line with 2+ pipes is treated as a table row (markdown OR loose "a | b | c").
    return line.count("|") >= 2


def _split_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [_latin1(p) for p in parts]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _draw_table(pdf: FPDF, rows: list[list[str]], heading: bool) -> None:
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]  # pad ragged rows
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(borders_layout="ALL", text_align="LEFT", first_row_as_headings=heading) as table:
        for r in rows:
            row = table.row()
            for cell in r:
                row.cell(cell)
    pdf.ln(2)


def _render_body(pdf: FPDF, text: str) -> None:
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            block, had_sep = [], False
            while i < n and _is_table_line(lines[i]):
                if _is_separator(lines[i]):
                    had_sep = True
                else:
                    block.append(_split_row(lines[i]))
                i += 1
            _draw_table(pdf, block, heading=had_sep)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, _latin1(lines[i]) if lines[i].strip() else " ",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            i += 1


def build_pdf(title: str, chart: list[tuple[str, float]], body_text: str) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _latin1(title or "Smile Export"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
        _render_body(pdf, body_text)

    return bytes(pdf.output())
