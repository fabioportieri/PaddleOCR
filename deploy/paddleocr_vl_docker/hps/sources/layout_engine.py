from dataclasses import dataclass
from typing import List, Dict
import math


@dataclass
class Column:
    lines: List
    x1: int
    x2: int


class LayoutEngine:
    """
    Responsible ONLY for spatial layout:
    - column detection
    - reading order reconstruction
    - optional line fixing
    """

    def __init__(self, column_gap_threshold: int = 80):
        self.column_gap_threshold = column_gap_threshold

    # =========================
    # Public API
    # =========================

    def organize(self, areas: List) -> List:
        """
        Input: unordered areas (with lines inside)
        Output: reordered + column-grouped areas
        """

        all_lines = self._flatten_lines(areas)

        columns = self._detect_columns(all_lines)

        ordered_lines = self._order_columns(columns)

        return self._rebuild_areas(ordered_lines, areas)

    # =========================
    # Column detection
    # =========================

    def _detect_columns(self, lines: List) -> List[Column]:
        """
        Simple heuristic clustering based on x-centers.
        """

        if not lines:
            return []

        # compute x-centers
        enriched = [
            (line, (line.bbox.x1 + line.bbox.x2) / 2)
            for line in lines
        ]

        enriched.sort(key=lambda x: x[1])

        columns: List[Column] = []

        for line, cx in enriched:
            placed = False

            for col in columns:
                if abs(cx - col_center(col)) < self.column_gap_threshold:
                    col.lines.append(line)
                    col.x1 = min(col.x1, line.bbox.x1)
                    col.x2 = max(col.x2, line.bbox.x2)
                    placed = True
                    break

            if not placed:
                columns.append(Column(
                    lines=[line],
                    x1=line.bbox.x1,
                    x2=line.bbox.x2
                ))

        return columns

    # =========================
    # Reading order
    # =========================

    def _order_columns(self, columns: List[Column]) -> List:
        """
        Top-to-bottom inside columns, left-to-right across columns
        """

        columns.sort(key=lambda c: c.x1)

        ordered = []

        for col in columns:
            col.lines.sort(key=lambda l: l.bbox.y1)
            ordered.extend(col.lines)

        return ordered

    # =========================
    # Reconstruction layer
    # =========================

    def fix_broken_lines(self, lines: List) -> List:
        """
        Optional: merge lines that are likely split incorrectly.
        """

        if not lines:
            return lines

        fixed = [lines[0]]

        for i in range(1, len(lines)):
            prev = fixed[-1]
            curr = lines[i]

            if self._should_merge(prev, curr):
                fixed[-1] = self._merge_lines(prev, curr)
            else:
                fixed.append(curr)

        return fixed

    def _should_merge(self, a, b) -> bool:
        vertical_gap = b.bbox.y1 - a.bbox.y2
        if vertical_gap > 25:
            return False

        # heuristic: weak punctuation end OR low width break
        if len(a.words) > 0:
            last_word = a.words[-1].text
            if last_word.endswith(("-", ",")):
                return True

        return False

    def _merge_lines(self, a, b):
        merged_words = a.words + b.words
        merged_words.sort(key=lambda w: w.bbox.x1)

        bbox = BBox(
            min(a.bbox.x1, b.bbox.x1),
            min(a.bbox.y1, b.bbox.y1),
            max(a.bbox.x2, b.bbox.x2),
            max(a.bbox.y2, b.bbox.y2),
        )

        return type(a)(
            words=merged_words,
            bbox=bbox,
            id=a.id
        )

    # =========================
    # Helpers
    # =========================

    def _flatten_lines(self, areas: List) -> List:
        lines = []
        for a in areas:
            for p in a.paragraphs:
                lines.extend(p.lines)
        return lines


def col_center(col: Column) -> float:
    return (col.x1 + col.x2) / 2