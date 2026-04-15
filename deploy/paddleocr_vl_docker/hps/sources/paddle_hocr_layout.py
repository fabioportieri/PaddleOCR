from typing import List
from dataclasses import dataclass
import html

from paddlex_adapter import Box, LayoutBlock, OCRToken, NormalizedPage, to_box, paddlex_to_normalized

# -----------------------------
# Geometry helpers
# -----------------------------

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_center_y(b):
    return (b[1] + b[3]) / 2


def bbox_center_x(b):
    return (b[0] + b[2]) / 2


def overlap_y(a, b, thresh=10):
    return abs(bbox_center_y(a) - bbox_center_y(b)) < thresh


@dataclass
class HocrContext:
    doc_json: dict
    image_name: str | None
    page_id: int = 1


def _ir(v) -> int:
    """Round float coordinate to int."""
    return int(round(v))


def _bbox_str(b) -> str:
    return f"{_ir(b[0])} {_ir(b[1])} {_ir(b[2])} {_ir(b[3])}"


def _build_lines(words):
    """Cluster word dicts into text lines by y-proximity."""
    lines = []
    used = set()
    for w in words:
        if w["id"] in used:
            continue
        line = [w]
        used.add(w["id"])
        for w2 in words:
            if w2["id"] not in used and overlap_y(w["bbox"], w2["bbox"]):
                line.append(w2)
                used.add(w2["id"])
        line.sort(key=lambda x: x["bbox"][0])
        lines.append(line)
    return lines


def _render_carea(words, page_id, carea_id, block_bbox_str):
    lines = _build_lines(words)
    line_html = []
    for line_id, line in enumerate(lines, 1):
        line_bbox = [
            min(w["bbox"][0] for w in line),
            min(w["bbox"][1] for w in line),
            max(w["bbox"][2] for w in line),
            max(w["bbox"][3] for w in line),
        ]
        word_html = []
        for word_id, w in enumerate(line, 1):
            wx1, wy1, wx2, wy2 = w["bbox"]
            conf = int(w["score"] * 100)
            word_html.append(
                f'<span class="ocrx_word"'
                f' id="word_{page_id}_{carea_id}_{line_id}_{word_id}"'
                f' title="bbox {_ir(wx1)} {_ir(wy1)} {_ir(wx2)} {_ir(wy2)}; x_wconf {conf}">'
                f'{html.escape(w["text"])}</span>'
            )
        line_html.append(
            f'<span class="ocr_line"'
            f' id="line_{page_id}_{carea_id}_{line_id}"'
            f' title="bbox {_bbox_str(line_bbox)}">'
            f'{" ".join(word_html)}</span>'
        )
    return (
        f'<div class="ocr_carea"'
        f' id="block_{page_id}_{carea_id}"'
        f' title="bbox {block_bbox_str}">'
        f'<p class="ocr_par" id="par_{page_id}_{carea_id}_1">'
        f'{" ".join(line_html)}</p></div>'
    )


# -----------------------------
# Core HOCR builder
# -----------------------------

def paddlex_to_hocr_layout(page: NormalizedPage, page_id=1, image_name: str | None = None) -> str:
    layout = page.layout      # List[LayoutBlock]
    ocr_tokens = page.ocr     # List[OCRToken]

    # ---------------------------------
    # Build word list from OCRToken list
    # ---------------------------------
    page_words = []
    for i, token in enumerate(ocr_tokens):
        if not token.text:
            continue
        page_words.append({
            "id": i + 1,
            "text": token.text,
            "score": token.score,
            "bbox": [token.box.x1, token.box.y1, token.box.x2, token.box.y2],
            "matched": False,
        })

    if not page_words:
        return "<html><body><div class='ocr_page'></div></body></html>"

    # sort reading order (top-to-bottom, left-to-right)
    page_words.sort(key=lambda w: (w["bbox"][1], w["bbox"][0]))

    # -------------------------------------------------
    # Page bbox: derive from actual word coordinates
    # -------------------------------------------------
    page_bbox = [
        min(w["bbox"][0] for w in page_words),
        min(w["bbox"][1] for w in page_words),
        max(w["bbox"][2] for w in page_words),
        max(w["bbox"][3] for w in page_words),
    ]

    html_parts = []
    carea_id = 1

    # ----------------------------------------
    # One OCR_CAREA per detected layout block
    # ----------------------------------------
    for block in layout:
        bx1, by1, bx2, by2 = block.box.x1, block.box.y1, block.box.x2, block.box.y2

        block_words = [
            w for w in page_words
            if bx1 <= bbox_center_x(w["bbox"]) <= bx2
            and by1 <= bbox_center_y(w["bbox"]) <= by2
        ]

        if not block_words:
            continue

        for w in block_words:
            w["matched"] = True

        html_parts.append(
            _render_carea(
                block_words, page_id, carea_id,
                f"{_ir(bx1)} {_ir(by1)} {_ir(bx2)} {_ir(by2)}"
            )
        )
        carea_id += 1

    # -------------------------------------------------------
    # Fallback: words whose center fell outside every block
    # -------------------------------------------------------
    unmatched = [w for w in page_words if not w["matched"]]
    if unmatched:
        ub = [
            min(w["bbox"][0] for w in unmatched),
            min(w["bbox"][1] for w in unmatched),
            max(w["bbox"][2] for w in unmatched),
            max(w["bbox"][3] for w in unmatched),
        ]
        html_parts.append(
            _render_carea(unmatched, page_id, carea_id, _bbox_str(ub))
        )

    # ------------------
    # Final HOCR output
    # ------------------
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'
        ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        '<head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>'
        '<meta name="ocr-system" content="PaddleOCR PaddleX"/>'
        '<meta name="ocr-capabilities"'
        ' content="ocr_page ocr_carea ocr_par ocr_line ocrx_word"/>'
        '</head><body>'
        f'<div class="ocr_page" id="page_{page_id}"'
        f' title="image {image_name}; bbox {_bbox_str(page_bbox)}; ppageno {page_id - 1}">'
        f'{"" .join(html_parts)}'
        '</div></body></html>'
    )