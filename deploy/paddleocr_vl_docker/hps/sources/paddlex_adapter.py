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
    """
    Supports:1
    - [x1,y1,x2,y2]
    - [x,y,w,h]
    """
    if len(b) == 4:
        x1, y1, a, b2 = b

        # heuristic: if a/b2 look like width/height
        if a > x1 or b2 > y1:
            return Box(x1, y1, a, b2)
        else:
            return Box(x1, y1, x1 + a, y1 + b2)

    raise ValueError(f"Unsupported box format: {b}")


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
            # derive table bbox from union of cell boxes
            xs1 = [b[0] for b in cell_boxes]
            ys1 = [b[1] for b in cell_boxes]
            xs2 = [b[2] for b in cell_boxes]
            ys2 = [b[3] for b in cell_boxes]
            table_boxes.append(Box(min(xs1), min(ys1), max(xs2), max(ys2)))
        table_tokens.extend(_tokens_from_ocr_pred(table.get("table_ocr_pred") or {}))

    # Overall OCR: keep only tokens whose center is NOT inside any table area
    overall_tokens = []
    for t in _tokens_from_ocr_pred(doc.get("overall_ocr_res") or {}):
        cx, cy = _center(t.box)
        if not any(_inside(cx, cy, tb) for tb in table_boxes):
            overall_tokens.append(t)

    return overall_tokens + table_tokens