# paddlex_adapter.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class LayoutBlock:
    type: str          # text / table / title / figure
    box: Box
    raw: Dict[str, Any]


@dataclass
class OCRToken:
    text: str
    box: Box
    score: float
    raw: Dict[str, Any]


@dataclass
class NormalizedPage:
    layout: List[LayoutBlock]
    ocr: List[OCRToken]
    raw: Dict[str, Any]
    width: int = 0
    height: int = 0

def to_box(b) -> Box:
    # PaddleX 3.x standard is [x1, y1, x2, y2]
    return Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))



def paddlex_to_normalized(doc_result: Any) -> NormalizedPage:
    """
    Accepts ANY PaddleX output shape safely.
    """

    # unwrap generator / object / dict
    if hasattr(doc_result, "json"):
        doc = doc_result.json
    else:
        doc = doc_result

    # handle list wrapper cases
    if isinstance(doc, list):
        doc = doc[0]

    # actual PaddleX output wraps everything under "res"
    if "res" in doc:
        doc = doc["res"]
    elif "results" in doc:
        doc = doc["results"][0].get("res", doc)

    return NormalizedPage(
        layout=extract_layout(doc),
        ocr=extract_ocr(doc),
        raw=doc,
        width=int(doc.get("width", 0)),
        height=int(doc.get("height", 0)),
    )

def extract_layout(doc: Dict[str, Any]) -> List[LayoutBlock]:
    layout = doc.get("layout_det_res", {}) or {}

    boxes = layout.get("boxes", []) or []

    out = []

    for b in boxes:
        # each box is a dict: {"cls_id": 0, "label": "", "score": 0.0, "coordinate": [x1,y1,x2,y2]}
        coord = b.get("coordinate", [0, 0, 0, 0]) if isinstance(b, dict) else b
        label = b.get("label", "unknown") if isinstance(b, dict) else "unknown"
        out.append(
            LayoutBlock(
                type=label,
                box=to_box(coord),
                raw=b
            )
        )

    return out

def _tokens_from_ocr_pred(ocr_pred: Dict[str, Any]) -> List[OCRToken]:
    """Extract OCRToken list from any dict that has rec_texts/rec_boxes/rec_scores."""
    texts = ocr_pred.get("rec_texts", []) or []
    boxes = ocr_pred.get("rec_boxes", []) or []
    scores = ocr_pred.get("rec_scores", []) or []
    tokens = []
    for i, text in enumerate(texts):
        if not text:
            continue
        box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
        score = scores[i] if i < len(scores) else 0.0
        tokens.append(OCRToken(text=text, box=to_box(box), score=score, raw={"box": box}))
    return tokens


def _center(box: Box):
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def _inside(cx, cy, b: Box) -> bool:
    return b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2


def extract_ocr(doc: Dict[str, Any]) -> List[OCRToken]:
    # Collect table bboxes so we can exclude overall_ocr duplicates
    table_boxes: List[Box] = []
    table_tokens: List[OCRToken] = []
    for table in (doc.get("table_res_list") or []):
        cell_boxes = table.get("cell_box_list") or []
        if cell_boxes:
            # derive table bbox from union of cell boxes (these are absolute page coords)
            xs1 = [b[0] for b in cell_boxes]
            ys1 = [b[1] for b in cell_boxes]
            xs2 = [b[2] for b in cell_boxes]
            ys2 = [b[3] for b in cell_boxes]
            table_box = Box(min(xs1), min(ys1), max(xs2), max(ys2))
            table_boxes.append(table_box)
        table_tokens.extend(_tokens_from_ocr_pred(table.get("table_ocr_pred") or {}))

    # Overall OCR: keep only tokens whose center is NOT inside any table area
    overall_tokens = []
    for t in _tokens_from_ocr_pred(doc.get("overall_ocr_res") or {}):
        cx, cy = _center(t.box)
        if not any(_inside(cx, cy, tb) for tb in table_boxes):
            overall_tokens.append(t)

    # parsing_res_list: non-table blocks (figure_title, header, text, number, footer, etc.)
    # These are section headings and labels not covered by overall_ocr or table_ocr.
    # Add them only when no overall_ocr token already falls inside the block bbox.
    parsing_tokens: List[OCRToken] = []
    for block in (doc.get("parsing_res_list") or []):
        label = block.get("block_label", "")
        if label == "table":
            continue
        content = (block.get("block_content") or "").strip()
        if not content:
            continue
        bbox = block.get("block_bbox") or []
        if len(bbox) != 4:
            continue
        bx = Box(bbox[0], bbox[1], bbox[2], bbox[3])
        # Skip if any overall_ocr token is already inside this block
        already_covered = any(
            _inside(_center(t.box)[0], _center(t.box)[1], bx)
            for t in overall_tokens
        )
        if not already_covered:
            parsing_tokens.append(OCRToken(text=content, box=bx, score=1.0, raw=block))

    return overall_tokens + parsing_tokens + table_tokens