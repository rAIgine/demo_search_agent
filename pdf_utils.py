# pdf_utils.py
import io
import re
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def _clean_text(text: str) -> str:
    """
    Normalise characters that are not well-supported by built-in Type 1 fonts.
    Kita tetap pakai Helvetica / Times / Courier (tanpa TTF),
    jadi aman kalau unicode 'aneh' di-map ke ASCII.
    """
    replacements = {
        "–": "-",   # en dash
        "—": "-",   # em dash
        "•": "-",   # bullet
        "·": "-",   # middle dot
        "\u00a0": " ",  # non-breaking space
        "\u2011": "-",  # non-breaking hyphen (macro-driven, dll)
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def _convert_markdown_bold_to_html(text: str) -> str:
    """
    Convert **bold** segments in markdown into <b>bold</b> untuk Paragraph.
    Paragraph ReportLab support subset HTML: <b>, <i>, <u>, <font>, <br/>, dll.
    """
    # Replace **text** with <b>text</b>
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _is_markdown_table(block: str) -> bool:
    """
    Heuristik: anggap block sebagai markdown table jika
    ada minimal 2 baris non-separator yang mulai dengan '|'.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [ln for ln in lines if ln.startswith("|")]
    if len(pipe_lines) < 2:
        return False
    # cek ada baris data (bukan cuma |---|)
    for ln in pipe_lines:
        inner = ln.strip("|").strip()
        if not inner:
            continue
        if set(inner) <= set("-:"):  # separator
            continue
        return True
    return False


def _parse_markdown_table(block: str) -> List[List[str]]:
    """
    Parse markdown table gaya GitHub jadi list 2D cell string.
    """
    rows: List[List[str]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # skip baris separator alignment (|---|:---:|---|)
        inner = "".join(cells)
        if inner and set(inner) <= set("-:"):
            continue
        rows.append(cells)
    return rows


def build_pdf_bytes(
    markdown_text: str,
    title: str | None = None,
    *,
    # 1.0 = single, 1.5 = satu setengah, 2.0 = double
    line_spacing: float = 1.5,
    # jarak sebelum/after paragraf dalam "jumlah baris"
    paragraph_space_before: float = 0.0,
    paragraph_space_after: float = 0.75,
) -> bytes:
    """
    Convert jawaban LLM (markdown-ish) jadi PDF rapi.

    Parameter utama:
    - line_spacing:
        1.0   -> mirip single
        1.5   -> 1.5 line
        2.0   -> double
    - paragraph_space_before / paragraph_space_after:
        diukur dalam "jumlah baris".
        Misal font 10pt:
          paragraph_space_after = 0.5 -> ~5pt di bawah paragraf

    Fitur konten:
    - Heading: '#', '##', '###' di awal baris
    - Bullet list: baris mulai '- ' atau '* '
    - Tabel markdown: '| col1 | col2 |'
    - **bold** → <b>bold</b>
    - Tabel → Table + TableStyle (header abu-abu, grid, striping).
    - Blok 'Summary', 'Ringkasan', 'Recommendation' → highlight.
    """
    # 1) Preprocess markdown: bold -> <b>..</b>, lalu normalisasi karakter
    text = _convert_markdown_bold_to_html(markdown_text or "")
    text = _clean_text(text)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # kita pakai ukuran font body sebagai dasar semua spacing
    body_font_size = 12.0
    body_leading = body_font_size * line_spacing
    space_before_pts = body_font_size * paragraph_space_before
    space_after_pts = body_font_size * paragraph_space_after

    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=body_font_size,
        leading=body_leading,
        spaceBefore=space_before_pts,
        spaceAfter=space_after_pts,
    )

    heading1 = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=16 * line_spacing,
        spaceBefore=body_font_size * (paragraph_space_before + 0.5),
        spaceAfter=body_font_size * (paragraph_space_after + 0.5),
    )

    heading2 = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=13 * line_spacing,
        spaceBefore=body_font_size * (paragraph_space_before + 0.25),
        spaceAfter=body_font_size * (paragraph_space_after + 0.5),
    )

    heading3 = ParagraphStyle(
        "Heading3Custom",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=11 * line_spacing,
        spaceBefore=body_font_size * paragraph_space_before,
        spaceAfter=body_font_size * paragraph_space_after,
    )

    heading4 = ParagraphStyle(
        "Heading4Custom",
        parent=styles["Heading4"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=11 * line_spacing,
        spaceBefore=body_font_size * paragraph_space_before,
        spaceAfter=body_font_size * paragraph_space_after,
    )

    summary_style = ParagraphStyle(
        "Summary",
        parent=body,
        fontName="Helvetica-Bold",
        backColor=colors.HexColor("#ecfeff"),
        textColor=colors.HexColor("#0f172a"),
        leftIndent=4,
        borderPadding=6,
        spaceBefore=body_font_size * (paragraph_space_before + 0.25),
        spaceAfter=body_font_size * (paragraph_space_after + 0.75),
    )

    story = []

    # Optional title di atas halaman pertama
    if title:
        story.append(Paragraph(_clean_text(title), heading1))

    # 2) Pecah jadi blok berdasarkan double newline
    blocks = text.split("\n\n")

    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue

        # ---------- TABLE ----------
        if _is_markdown_table(block):
            data = _parse_markdown_table(block)
            if not data:
                continue

            # Wrap cell dengan Paragraph supaya text bisa wrap
            header_style = ParagraphStyle(
                "TableHeader",
                parent=body,
                fontName="Helvetica-Bold",
                alignment=1,  # center
                spaceBefore=0,
                spaceAfter=0,
            )
            cell_style = ParagraphStyle(
                "TableCell",
                parent=body,
                alignment=1,  # center
                spaceBefore=0,
                spaceAfter=0,
            )

            table_data: List[List[Paragraph]] = []
            for row_idx, row in enumerate(data):
                row_cells: List[Paragraph] = []
                for cell in row:
                    cell_text = cell or ""
                    if row_idx == 0:
                        row_cells.append(Paragraph(cell_text, header_style))
                    else:
                        row_cells.append(Paragraph(cell_text, cell_style))
                table_data.append(row_cells)

            tbl = Table(table_data, hAlign="LEFT")
            tbl_style = TableStyle(
                [
                    # header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 4),
                    # grid & striping
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")]),
                ]
            )
            tbl.setStyle(tbl_style)

            story.append(tbl)
            # extra space setelah tabel
            story.append(Spacer(1, body_font_size * paragraph_space_after))
            continue

        # ---------- HEADINGS ----------
        if block.startswith("#### "):
            txt = block[4:].strip()
            story.append(Paragraph(txt, heading4))
            continue

        if block.startswith("### "):
            txt = block[4:].strip()
            story.append(Paragraph(txt, heading3))
            continue

        if block.startswith("## "):
            txt = block[3:].strip()
            story.append(Paragraph(txt, heading2))
            continue

        if block.startswith("# "):
            txt = block[2:].strip()
            story.append(Paragraph(txt, heading1))
            continue

        lower = block.lower()

        # ---------- HIGHLIGHT SUMMARY / RECOMMENDATION ----------
        if (
            lower.startswith("summary")
            or lower.startswith("ringkasan")
            or lower.startswith("recommendation")
        ):
            lines = [ln.rstrip() for ln in block.splitlines()]
            html_text = "<br/>".join(lines)
            story.append(Paragraph(html_text, summary_style))
            continue

        # ---------- NORMAL PARAGRAPH & BULLETS ----------
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append("• " + stripped[2:].strip())
            elif stripped.startswith("* "):
                lines.append("• " + stripped[2:].strip())
            else:
                lines.append(line.rstrip())

        html_text = "<br/>".join(lines)
        story.append(Paragraph(html_text, body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
