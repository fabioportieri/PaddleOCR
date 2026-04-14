from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_LABELS = {
    "text",
    "content",
    "paragraph_title",
    "doc_title",
    "abstract_title",
    "reference_title",
    "content_title",
    "table_title",
    "figure_title",
    "chart_title",
    "abstract",
    "reference",
    "reference_content",
    "algorithm",
    "number",
    "footnote",
    "header",
    "footer",
    "aside_text",
    "vertical_text",
    "vision_footnote",
    "ocr",
}


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_hocr(self) -> str:
        return f"bbox {self.x1} {self.y1} {self.x2} {self.y2}"

    @staticmethod
    def from_polygon(poly: Sequence[Sequence[float]]) -> "BBox":
        pts = _as_points(poly)
        xs = [p[0] for p in pts]
        x_min = int(min(xs))
        x_max = int(max(xs))

        if len(pts) == 4:
            y_min = int((pts[0][1] + pts[1][1]) / 2)
            y_max = int((pts[2][1] + pts[3][1]) / 2)
        else:
            ys = [p[1] for p in pts]
            y_min = int(min(ys))
            y_max = int(max(ys))

        return BBox(x_min, y_min, x_max, y_max)


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: Optional[int] = None
    wid: str = ""

    def to_html(self) -> str:
        conf_part = f"; x_wconf {self.conf}" if self.conf is not None else ""
        return (
            f'<span class="ocrx_word" id="{self.wid}" '
            f'title="{self.bbox.as_hocr()}{conf_part}">{escape(self.text)}</span>'
        )


@dataclass
class Line:
    words: List[Word]
    bbox: BBox
    conf: Optional[int] = None
    lid: str = ""

    def to_html(self) -> str:
        conf = self.conf if self.conf is not None else 95
        words_html = " ".join(word.to_html() for word in self.words)
        return (
            f'<span class="ocr_line" id="{self.lid}" '
            f'title="{self.bbox.as_hocr()}; baseline 0 0; x_wconf {conf}">{words_html}</span>'
        )


@dataclass
class Paragraph:
    lines: List[Line]
    bbox: BBox
    pid: str = ""
    lang: str = "eng"

    def to_html(self) -> str:
        lines_html = "\n".join(line.to_html() for line in self.lines)
        return (
            f'<p class="ocr_par" id="{self.pid}" lang="{self.lang}" '
            f'title="{self.bbox.as_hocr()}">\n{lines_html}\n</p>'
        )


@dataclass
class Area:
    paragraphs: List[Paragraph]
    bbox: BBox
    aid: str = ""

    def to_html(self) -> str:
        pars_html = "\n".join(par.to_html() for par in self.paragraphs)
        return f'<div class="ocr_carea" id="{self.aid}" title="{self.bbox.as_hocr()}">\n{pars_html}\n</div>'


@dataclass
class Page:
    areas: List[Area]
    width: int
    height: int
    pid: str = "page_1"

    def to_html(self, ocr_system: str) -> str:
        areas_html = "\n".join(area.to_html() for area in self.areas)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n'
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n'
            '<head>\n'
            '<title></title>\n'
            '<meta http-equiv="content-type" content="text/html; charset=utf-8" />\n'
            f'<meta name="ocr-system" content="{escape(ocr_system)}" />\n'
            '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />\n'
            '</head>\n'
            '<body>\n'
            f'<div class="ocr_page" id="{self.pid}" title="bbox 0 0 {self.width} {self.height}">\n'
            f'{areas_html}\n'
            '</div>\n'
            '</body>\n'
            '</html>'
        )


class PaddleHOCRBuilder:
    def __init__(self) -> None:
        self.word_counter = 0
        self.line_counter = 0
        self.par_counter = 0
        self.area_counter = 0

    def build_page(self, result: Any, image_width: int, image_height: int, lang: str = "eng") -> Tuple[Page, List[str], str]:
        page_result = self._normalize_page_result(result)

        if self._has_spotting(page_result):
            areas, all_text = self._build_from_vl_spotting(page_result, lang)
            ocr_system = "PaddleOCR-VL"
        elif self._has_parsing_list(page_result):
            areas, all_text = self._build_from_vl_parsing(page_result, lang)
            ocr_system = "PaddleOCR-VL"
        elif self._has_classic_lines(page_result):
            areas, all_text = self._build_from_classic(page_result, lang)
            ocr_system = "PaddleOCR"
        elif isinstance(page_result, dict) and page_result.get("blocks"):
            areas, all_text = self._build_from_block_json(page_result, lang)
            ocr_system = "PaddleOCR"
        elif self._has_document_children(page_result):
            areas, all_text = self._build_from_document_tree(page_result, lang)
            ocr_system = "PaddleOCR"
        else:
            areas, all_text = [], []
            ocr_system = "PaddleOCR"

        page = Page(areas=areas, width=image_width, height=image_height)
        return page, all_text, ocr_system

    def _next_id(self, prefix: str, attr: str) -> str:
        val = getattr(self, attr) + 1
        setattr(self, attr, val)
        return f"{prefix}_{val}"

    def _normalize_page_result(self, result: Any) -> Any:
        if isinstance(result, list):
            return result[0] if result else {}
        return result or {}

    def _has_spotting(self, page: Any) -> bool:
        spotting = page.get("spotting_res") if isinstance(page, dict) else None
        return bool(spotting and spotting.get("rec_polys") and spotting.get("rec_texts"))

    def _has_parsing_list(self, page: Any) -> bool:
        if not isinstance(page, dict):
            return False
        parsing = page.get("parsing_res_list")
        return isinstance(parsing, list) and len(parsing) > 0

    def _has_classic_lines(self, page: Any) -> bool:
        if not isinstance(page, dict):
            return False
        return isinstance(page.get("rec_texts"), list) and isinstance(page.get("rec_polys"), list)

    def _has_document_children(self, page: Any) -> bool:
        if not isinstance(page, dict):
            return False
        children = page.get("children")
        return isinstance(children, list) and len(children) > 0

    def _build_from_vl_spotting(self, page: Dict[str, Any], lang: str) -> Tuple[List[Area], List[str]]:
        spotting = page.get("spotting_res", {})
        rec_polys = spotting.get("rec_polys", [])
        rec_texts = spotting.get("rec_texts", [])

        word_boxes = [
            (str(txt), poly)
            for txt, poly in zip(rec_texts, rec_polys)
            if str(txt).strip()
        ]

        grouped_lines = _group_words_into_lines(word_boxes)

        areas: List[Area] = []
        all_text: List[str] = []

        for line_words in grouped_lines:
            words: List[Word] = []
            for text, poly in line_words:
                bbox = BBox.from_polygon(poly)
                word = Word(
                    text=text,
                    bbox=bbox,
                    conf=95,
                    wid=self._next_id("word", "word_counter"),
                )
                words.append(word)

            if not words:
                continue

            line_bbox = _merge_bbox([w.bbox for w in words])
            line_text = " ".join(w.text for w in words)
            all_text.append(line_text)

            line = Line(
                words=words,
                bbox=line_bbox,
                conf=95,
                lid=self._next_id("line", "line_counter"),
            )
            par = Paragraph(
                lines=[line],
                bbox=line_bbox,
                pid=self._next_id("par", "par_counter"),
                lang=lang,
            )
            area = Area(
                paragraphs=[par],
                bbox=line_bbox,
                aid=self._next_id("carea", "area_counter"),
            )
            areas.append(area)

        return areas, all_text

    def _build_from_vl_parsing(self, page: Dict[str, Any], lang: str) -> Tuple[List[Area], List[str]]:
        parsing = page.get("parsing_res_list") or []

        areas: List[Area] = []
        all_text: List[str] = []

        for block in parsing:
            label = _obj_get(block, "label") or _obj_get(block, "block_label") or ""
            content = _obj_get(block, "content") or _obj_get(block, "block_content") or ""
            bbox_data = _obj_get(block, "bbox") or _obj_get(block, "block_bbox")

            if not str(content).strip() or label not in TEXT_LABELS:
                continue
            if not bbox_data or len(bbox_data) < 4:
                continue

            bx0, by0, bx1, by1 = [int(v) for v in bbox_data[:4]]
            block_bbox = BBox(bx0, by0, bx1, by1)

            text_lines = [line.strip() for line in str(content).split("\n") if line.strip()]
            if not text_lines:
                continue

            block_h = max(by1 - by0, 1)
            line_h = max(block_h // len(text_lines), 1)

            lines: List[Line] = []
            for idx, line_text in enumerate(text_lines):
                ly0 = by0 + idx * line_h
                ly1 = min(by0 + (idx + 1) * line_h, by1)
                line_bbox = BBox(bx0, ly0, bx1, ly1)

                words = _estimate_words_for_line(
                    line_text=line_text,
                    line_bbox=line_bbox,
                    confidence=95,
                    id_factory=lambda: self._next_id("word", "word_counter"),
                )
                if not words:
                    continue

                lines.append(
                    Line(
                        words=words,
                        bbox=line_bbox,
                        conf=95,
                        lid=self._next_id("line", "line_counter"),
                    )
                )
                all_text.append(line_text)

            if not lines:
                continue

            par = Paragraph(
                lines=lines,
                bbox=block_bbox,
                pid=self._next_id("par", "par_counter"),
                lang=lang,
            )
            areas.append(
                Area(
                    paragraphs=[par],
                    bbox=block_bbox,
                    aid=self._next_id("carea", "area_counter"),
                )
            )

        return areas, all_text

    def _build_from_classic(self, page: Dict[str, Any], lang: str) -> Tuple[List[Area], List[str]]:
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys") or []
        text_words = page.get("text_word") or []
        text_word_regions = page.get("text_word_region") or []

        has_word_boxes = bool(text_words and text_word_regions)

        areas: List[Area] = []
        all_text: List[str] = []

        for idx, (text, score, poly) in enumerate(zip(texts, scores, polys)):
            text_str = str(text)
            if not text_str.strip():
                continue

            line_bbox = BBox.from_polygon(poly)
            conf = int(float(score) * 100) if score is not None else 95
            all_text.append(text_str)

            if (
                has_word_boxes
                and idx < len(text_words)
                and idx < len(text_word_regions)
                and text_words[idx]
                and text_word_regions[idx]
            ):
                words = self._build_words_from_native_word_boxes(text_words[idx], text_word_regions[idx], conf)
            else:
                words = _estimate_words_for_line(
                    line_text=text_str,
                    line_bbox=line_bbox,
                    confidence=conf,
                    id_factory=lambda: self._next_id("word", "word_counter"),
                )

            if not words:
                continue

            line = Line(
                words=words,
                bbox=line_bbox,
                conf=conf,
                lid=self._next_id("line", "line_counter"),
            )
            par = Paragraph(
                lines=[line],
                bbox=line_bbox,
                pid=self._next_id("par", "par_counter"),
                lang=lang,
            )
            areas.append(
                Area(
                    paragraphs=[par],
                    bbox=line_bbox,
                    aid=self._next_id("carea", "area_counter"),
                )
            )

        return areas, all_text

    def _build_words_from_native_word_boxes(self, tokens: Sequence[Any], boxes: Sequence[Any], conf: int) -> List[Word]:
        merged_words: List[Tuple[str, List[Any]]] = []
        current_tokens: List[str] = []
        current_boxes: List[Any] = []

        for token, box in zip(tokens, boxes):
            token_str = str(token).strip()
            if not token_str:
                if current_tokens:
                    merged_words.append(("".join(current_tokens), current_boxes))
                    current_tokens = []
                    current_boxes = []
                continue
            current_tokens.append(token_str)
            current_boxes.append(box)

        if current_tokens:
            merged_words.append(("".join(current_tokens), current_boxes))

        words: List[Word] = []
        for text, sub_boxes in merged_words:
            sub_bboxes = [BBox.from_polygon(poly) for poly in sub_boxes if poly is not None]
            if not sub_bboxes:
                continue
            union = _merge_bbox(sub_bboxes)
            words.append(
                Word(
                    text=text,
                    bbox=union,
                    conf=conf,
                    wid=self._next_id("word", "word_counter"),
                )
            )

        words.sort(key=lambda w: w.bbox.x1)
        return words

    def _build_from_block_json(self, page: Dict[str, Any], lang: str) -> Tuple[List[Area], List[str]]:
        blocks = page.get("blocks", [])

        areas: List[Area] = []
        all_text: List[str] = []

        for block in blocks:
            if block.get("type") == "table":
                continue

            block_bbox = _bbox_from_any(block)
            raw_lines = block.get("lines", [])
            lines: List[Line] = []

            for raw_line in raw_lines:
                raw_words = raw_line.get("words", [])
                words: List[Word] = []
                for raw_word in raw_words:
                    wtext = str(raw_word.get("text", "")).strip()
                    if not wtext:
                        continue
                    wbbox = _bbox_from_any(raw_word)
                    conf_val = raw_word.get("confidence")
                    conf = int(conf_val) if conf_val is not None else None
                    words.append(
                        Word(
                            text=wtext,
                            bbox=wbbox,
                            conf=conf,
                            wid=self._next_id("word", "word_counter"),
                        )
                    )

                words.sort(key=lambda w: w.bbox.x1)
                if not words:
                    continue

                line_bbox = _merge_bbox([w.bbox for w in words])
                lines.append(
                    Line(
                        words=words,
                        bbox=line_bbox,
                        conf=95,
                        lid=self._next_id("line", "line_counter"),
                    )
                )
                all_text.append(" ".join(w.text for w in words))

            if not lines:
                continue

            par = Paragraph(
                lines=lines,
                bbox=block_bbox,
                pid=self._next_id("par", "par_counter"),
                lang=lang,
            )
            areas.append(
                Area(
                    paragraphs=[par],
                    bbox=block_bbox,
                    aid=self._next_id("carea", "area_counter"),
                )
            )

        areas.sort(key=lambda a: (a.bbox.y1, a.bbox.x1))
        return areas, all_text

    def _build_from_document_tree(self, page: Dict[str, Any], lang: str) -> Tuple[List[Area], List[str]]:
        areas: List[Area] = []
        all_text: List[str] = []

        for node in _iter_content_nodes(page):
            bbox = _bbox_from_any(node)
            if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
                continue

            lines_text = _extract_lines_from_node(node)
            if not lines_text:
                continue

            block_h = max(bbox.y2 - bbox.y1, 1)
            line_h = max(block_h // len(lines_text), 1)
            lines: List[Line] = []

            for idx, line_text in enumerate(lines_text):
                y0 = bbox.y1 + idx * line_h
                y1 = min(bbox.y1 + (idx + 1) * line_h, bbox.y2)
                line_bbox = BBox(bbox.x1, y0, bbox.x2, y1)
                words = _estimate_words_for_line(
                    line_text=line_text,
                    line_bbox=line_bbox,
                    confidence=95,
                    id_factory=lambda: self._next_id("word", "word_counter"),
                )
                if not words:
                    continue

                lines.append(
                    Line(
                        words=words,
                        bbox=line_bbox,
                        conf=95,
                        lid=self._next_id("line", "line_counter"),
                    )
                )
                all_text.append(line_text)

            if not lines:
                continue

            par = Paragraph(
                lines=lines,
                bbox=bbox,
                pid=self._next_id("par", "par_counter"),
                lang=lang,
            )
            areas.append(
                Area(
                    paragraphs=[par],
                    bbox=bbox,
                    aid=self._next_id("carea", "area_counter"),
                )
            )

        areas.sort(key=lambda a: (a.bbox.y1, a.bbox.x1))
        return areas, all_text


class HOCRRenderer:
    def render(self, page: Page, ocr_system: str) -> str:
        return page.to_html(ocr_system=ocr_system)


class PaddleHOCRConverter2:
    """Standalone Paddle JSON -> hOCR converter.

    This module intentionally has no OCRmyPDF plugin/runtime dependencies.
    Input should be a Paddle OCR/VL page result (or list whose first item is the page result).
    """

    def __init__(self) -> None:
        self.builder = PaddleHOCRBuilder()
        self.renderer = HOCRRenderer()

    def convert(self, result: Any, image_width: int, image_height: int, lang: str = "eng") -> Tuple[str, str]:
        page, all_text, ocr_system = self.builder.build_page(
            result=result,
            image_width=image_width,
            image_height=image_height,
            lang=lang,
        )
        hocr = self.renderer.render(page, ocr_system=ocr_system)
        text = "\n".join(all_text)
        return hocr, text


def convert_paddle_json_to_hocr(result: Any, image_width: int, image_height: int, lang: str = "eng") -> Tuple[str, str]:
    converter = PaddleHOCRConverter2()
    return converter.convert(result=result, image_width=image_width, image_height=image_height, lang=lang)


def _obj_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_points(poly: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    try:
        iterable = poly.tolist() if hasattr(poly, "tolist") else poly
    except Exception:
        iterable = poly

    if not isinstance(iterable, Iterable):
        return points

    for pt in iterable:
        try:
            p = pt.tolist() if hasattr(pt, "tolist") else pt
        except Exception:
            p = pt

        if isinstance(p, Sequence) and len(p) >= 2:
            points.append((float(p[0]), float(p[1])))

    return points


def _bbox_from_any(obj: Dict[str, Any]) -> BBox:
    bbox = obj.get("bbox")
    if bbox and len(bbox) >= 4:
        return BBox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    poly = obj.get("polygon", [])
    if not poly:
        return BBox(0, 0, 0, 0)
    return BBox.from_polygon(poly)


def _iter_content_nodes(root: Any) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if not isinstance(obj, dict):
            return

        block_type = str(obj.get("block_type", "")).lower()
        html_text = str(obj.get("html", "") or "")
        text = str(obj.get("text", "") or "")
        has_bbox = bool(obj.get("bbox") or obj.get("polygon"))

        is_content = has_bbox and (
            bool(text.strip())
            or bool(_normalize_html_text_for_check(html_text))
        )
        is_page_like = block_type in {"page", "document"}

        if is_content and not is_page_like:
            nodes.append(obj)

        children = obj.get("children")
        if isinstance(children, list):
            for child in children:
                walk(child)

    walk(root)
    return nodes


def _extract_lines_from_node(node: Dict[str, Any]) -> List[str]:
    text = str(node.get("text", "") or "")
    if text.strip():
        return _split_lines(text)

    html_text = str(node.get("html", "") or "")
    if not html_text.strip():
        return []

    if re.search(r"<tr\b", html_text, flags=re.IGNORECASE):
        return _extract_lines_from_table_html(html_text)

    return _extract_lines_from_generic_html(html_text)


def _extract_lines_from_table_html(html_text: str) -> List[str]:
    lines: List[str] = []
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", flags=re.IGNORECASE | re.DOTALL)

    for row_html in row_pattern.findall(html_text):
        cells = cell_pattern.findall(row_html)
        if not cells:
            line = _strip_html(row_html)
            if line:
                lines.append(line)
            continue
        parts = [_strip_html(cell) for cell in cells]
        parts = [p for p in parts if p]
        if parts:
            lines.append(" | ".join(parts))

    return _dedupe_empty(lines)


def _extract_lines_from_generic_html(html_text: str) -> List[str]:
    transformed = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
    transformed = re.sub(
        r"</(p|div|li|h1|h2|h3|h4|h5|h6|section|article|header|footer|caption)>",
        "\n",
        transformed,
        flags=re.IGNORECASE,
    )
    plain = _strip_html(transformed)
    return _split_lines(plain)


def _strip_html(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    decoded = unescape(no_tags)
    normalized = re.sub(r"\s+", " ", decoded).strip()
    return normalized


def _split_lines(value: str) -> List[str]:
    out = []
    for raw in value.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            out.append(line)
    return _dedupe_empty(out)


def _normalize_html_text_for_check(value: str) -> str:
    reduced = re.sub(r"<content-ref[^>]*>", "", value, flags=re.IGNORECASE)
    reduced = re.sub(r"</content-ref>", "", reduced, flags=re.IGNORECASE)
    return _strip_html(reduced)


def _dedupe_empty(lines: List[str]) -> List[str]:
    return [line for line in lines if line and line.strip()]


def _merge_bbox(boxes: Sequence[BBox]) -> BBox:
    return BBox(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def _group_words_into_lines(word_boxes: Sequence[Tuple[str, Any]]) -> List[List[Tuple[str, Any]]]:
    if not word_boxes:
        return []

    def poly_y_center(poly: Any) -> float:
        pts = _as_points(poly)
        if not pts:
            return 0.0
        return sum(p[1] for p in pts) / len(pts)

    def poly_height(poly: Any) -> float:
        pts = _as_points(poly)
        if len(pts) == 4:
            y_top = (pts[0][1] + pts[1][1]) / 2
            y_bot = (pts[2][1] + pts[3][1]) / 2
            return abs(y_bot - y_top)
        if not pts:
            return 0.0
        ys = [p[1] for p in pts]
        return abs(max(ys) - min(ys))

    sorted_words = sorted(word_boxes, key=lambda wb: poly_y_center(wb[1]))

    heights = sorted(poly_height(wb[1]) for wb in sorted_words)
    median_height = heights[len(heights) // 2] if heights else 20.0
    threshold = max(median_height * 0.6, 1.0)

    lines: List[List[Tuple[str, Any]]] = []
    current_line = [sorted_words[0]]
    current_y = poly_y_center(sorted_words[0][1])

    for text, poly in sorted_words[1:]:
        y_center = poly_y_center(poly)
        if abs(y_center - current_y) <= threshold:
            current_line.append((text, poly))
        else:
            lines.append(current_line)
            current_line = [(text, poly)]
            current_y = y_center

    lines.append(current_line)

    for idx, line in enumerate(lines):
        lines[idx] = sorted(line, key=lambda wb: BBox.from_polygon(wb[1]).x1)

    return lines


def _estimate_words_for_line(
    line_text: str,
    line_bbox: BBox,
    confidence: Optional[int],
    id_factory,
) -> List[Word]:
    words = [w for w in line_text.split() if w]
    if not words:
        return []

    line_w = max(line_bbox.x2 - line_bbox.x1, 1)
    total_chars = sum(len(word) for word in words)
    num_spaces = len(words) - 1

    if total_chars + num_spaces > 0:
        space_w = int((line_w * num_spaces) / (total_chars + num_spaces))
    else:
        space_w = 0
    word_area_w = line_w - space_w * num_spaces

    out: List[Word] = []
    cur_x = line_bbox.x1

    for idx, word in enumerate(words):
        if total_chars > 0:
            w_width = int(word_area_w * len(word) / total_chars)
        else:
            w_width = line_w // len(words)

        wx2 = line_bbox.x2 if idx == len(words) - 1 else cur_x + w_width

        out.append(
            Word(
                text=word,
                bbox=BBox(cur_x, line_bbox.y1, wx2, line_bbox.y2),
                conf=confidence,
                wid=id_factory(),
            )
        )
        cur_x = wx2 + space_w

    return out
