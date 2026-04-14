from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import html
from layout_engine import LayoutEngine

# =========================
# Models
# =========================

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_hocr(self) -> str:
        return f"bbox {self.x1} {self.y1} {self.x2} {self.y2}"


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: Optional[float] = None
    id: Optional[str] = None

    def to_html(self) -> str:
        conf_part = f"; x_wconf {int(self.conf)}" if self.conf is not None else ""
        return (
            f'<span class="ocrx_word" id="{self.id}" '
            f'title="{self.bbox.to_hocr()}{conf_part}">'
            f'{html.escape(self.text)}</span>'
        )


@dataclass
class Line:
    words: List[Word]
    bbox: BBox
    id: Optional[str] = None

    def to_html(self) -> str:
        words_html = " ".join(w.to_html() for w in self.words)
        return (
            f'<span class="ocr_line" id="{self.id}" '
            f'title="{self.bbox.to_hocr()}">{words_html}</span>'
        )


@dataclass
class Paragraph:
    lines: List[Line]
    bbox: BBox
    id: Optional[str] = None
    lang: str = "it"

    def to_html(self) -> str:
        lines_html = "\n".join(l.to_html() for l in self.lines)
        return (
            f'<p class="ocr_par" id="{self.id}" lang="{self.lang}" '
            f'title="{self.bbox.to_hocr()}">\n{lines_html}\n</p>'
        )


@dataclass
class Area:
    paragraphs: List[Paragraph]
    bbox: BBox
    id: Optional[str] = None

    def to_html(self) -> str:
        pars_html = "\n".join(p.to_html() for p in self.paragraphs)
        return (
            f'<div class="ocr_carea" id="{self.id}" '
            f'title="{self.bbox.to_hocr()}">\n{pars_html}\n</div>'
        )


@dataclass
class Page:
    areas: List[Area]
    bbox: BBox
    id: str = "page_1"
    page_no: int = 0

    def to_html(self) -> str:
        areas_html = "\n".join(a.to_html() for a in self.areas)
        return (
            f'<div class="ocr_page" id="{self.id}" '
            f'title="{self.bbox.to_hocr()}; ppageno {self.page_no}">\n'
            f'{areas_html}\n</div>'
        )


# =========================
# Builder (JSON → Model)
# =========================

class PaddleToHOCRBuilder:

    def __init__(self):
        self.word_counter = 0
        self.line_counter = 0
        self.par_counter = 0
        self.area_counter = 0

    def _next_id(self, prefix: str, counter_attr: str) -> str:
        val = getattr(self, counter_attr) + 1
        setattr(self, counter_attr, val)
        return f"{prefix}_{val}"

    def build(self, data: Dict[str, Any]) -> Page:
        page_bbox = self._extract_page_bbox(data)

        areas = []

        blocks = data.get("blocks", [])

        for block in blocks:
            if block.get("type") == "table":
                continue  # ignore tables per your spec

            area = self._build_area(block)
            if area:
                areas.append(area)

        # TODO: sort areas properly (column detection later)
        areas = sorted(areas, key=lambda a: (a.bbox.y1, a.bbox.x1))

        return Page(
            areas=areas,
            bbox=page_bbox
        )

    def _extract_page_bbox(self, data: Dict[str, Any]) -> BBox:
        # fallback safe
        return BBox(0, 0, data.get("width", 1000), data.get("height", 1000))

    def _build_area(self, block: Dict[str, Any]) -> Optional[Area]:
        bbox = self._bbox_from(block)

        lines = self._extract_lines(block)

        if not lines:
            return None

        paragraph = Paragraph(
            lines=lines,
            bbox=bbox,
            id=self._next_id("par", "par_counter")
        )

        return Area(
            paragraphs=[paragraph],
            bbox=bbox,
            id=self._next_id("block", "area_counter")
        )

    def _extract_lines(self, block: Dict[str, Any]) -> List[Line]:
        lines = []

        # Case 1: Paddle already gives lines
        raw_lines = block.get("lines", [])

        for l in raw_lines:
            words = self._extract_words(l)
            if not words:
                continue

            line_bbox = self._merge_bbox([w.bbox for w in words])

            line = Line(
                words=words,
                bbox=line_bbox,
                id=self._next_id("line", "line_counter")
            )

            lines.append(line)

        return lines

    def _extract_words(self, line: Dict[str, Any]) -> List[Word]:
        words = []

        for w in line.get("words", []):
            bbox = self._bbox_from(w)

            word = Word(
                text=w.get("text", ""),
                bbox=bbox,
                conf=w.get("confidence"),
                id=self._next_id("word", "word_counter")
            )
            words.append(word)

        # enforce left → right
        words.sort(key=lambda w: w.bbox.x1)

        return words

    def _bbox_from(self, obj: Dict[str, Any]) -> BBox:
        if "bbox" in obj:
            x1, y1, x2, y2 = obj["bbox"]
            return BBox(x1, y1, x2, y2)

        # polygon fallback
        poly = obj.get("polygon", [])
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        return BBox(min(xs), min(ys), max(xs), max(ys))

    def _merge_bbox(self, boxes: List[BBox]) -> BBox:
        return BBox(
            min(b.x1 for b in boxes),
            min(b.y1 for b in boxes),
            max(b.x2 for b in boxes),
            max(b.y2 for b in boxes),
        )


# =========================
# Renderer
# =========================

class HOCRRenderer:

    def render(self, page: Page) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>hOCR</title>
</head>
<body>
{page.to_html()}
</body>
</html>
"""


# =========================
# Facade (what you call)
# =========================

class PaddleToHOCRConverter:

    def __init__(self):
        self.builder = PaddleToHOCRBuilder()
        self.renderer = HOCRRenderer()
        self.layout = LayoutEngine()

    def convert(self, json_data: Dict[str, Any]) -> str:
        page = self.builder.build(json_data)

        #  apply layout normalization
        page.areas = self.layout.organize(page.areas)

        return self.renderer.render(page)