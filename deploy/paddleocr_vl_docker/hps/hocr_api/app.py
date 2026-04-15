from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import fastapi
import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse


LAYOUT_PARSING_URL = os.getenv(
    "HPS_LAYOUT_PARSING_URL", "http://paddleocr-vl-api:8080/layout-parsing"
)
REQUEST_TIMEOUT = int(os.getenv("HPS_LAYOUT_PARSING_TIMEOUT", "600"))
LOG_LEVEL = os.getenv("HPS_LOG_LEVEL", "INFO")
JSON_DUMP_DIR = os.getenv("HPS_JSON_DUMP_DIR", "/data/hocr-debug")

logger = logging.getLogger("hocr-wrapper")


def _configure_logger() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_configure_logger()


def _load_converter_class():
    converter_path = Path(__file__).resolve().parent.parent / "sources" / "paddle-hocr-converter2.py"
    spec = spec_from_file_location("paddle_hocr_converter2_dynamic", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter module from {converter_path}")

    module = module_from_spec(spec)
    # Ensure the module is registered before execution so dataclasses can resolve __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "PaddleHOCRConverter2"):
        raise RuntimeError("PaddleHOCRConverter2 not found in paddle-hocr-converter2.py")
    return module.PaddleHOCRConverter2


PaddleHOCRConverter2 = _load_converter_class()


app = fastapi.FastAPI(
    title="Paddle hOCR Wrapper API",
    description="Wraps /layout-parsing and returns hOCR HTML",
    version="1.0.0",
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/hocr", response_class=HTMLResponse)
async def hocr(request: Request, body: Dict[str, Any]) -> HTMLResponse:
    """Accept the same payload as /layout-parsing, call it, and return hOCR HTML."""
    request_log_id = body.get("logId", "")
    logger.info("Incoming /hocr request logId=%s", request_log_id)

    layout_json, status_code = await _call_layout_parsing(body)

    logger.info(
        "layout-parsing JSON response (status=%s): %s",
        status_code,
        json.dumps(layout_json, ensure_ascii=False),
    )

    if status_code >= 400:
        return JSONResponse(status_code=status_code, content=layout_json)

    if isinstance(layout_json, dict) and layout_json.get("errorCode", 0) not in (0, None):
        err_code = int(layout_json.get("errorCode", 500))
        return JSONResponse(status_code=err_code, content=layout_json)

    conversion_source = _extract_conversion_source(layout_json)
    _dump_request_json(
        request_log_id=request_log_id,
        request_body=body,
        layout_json=layout_json,
        conversion_source=conversion_source,
        status_code=status_code,
    )

    logger.info(
        "Selected conversion source summary: %s",
        _summarize_payload(conversion_source),
    )

    image_width, image_height = _extract_image_size(body, conversion_source)

    converter = PaddleHOCRConverter2()
    hocr_html, _text = converter.convert(
        result=conversion_source,
        image_width=image_width,
        image_height=image_height,
        lang="eng",
    )

    if 'class="ocr_carea"' not in hocr_html:
        logger.warning(
            "First conversion produced no carea blocks, retrying with full layout JSON"
        )
        image_width2, image_height2 = _extract_image_size(body, layout_json)
        hocr_html, _text = converter.convert(
            result=layout_json,
            image_width=image_width2,
            image_height=image_height2,
            lang="eng",
        )

    carea_count = hocr_html.count('class="ocr_carea"')
    logger.info("hOCR carea count=%s", carea_count)

    headers = {"Content-Disposition": 'inline; filename="result.hocr.html"'}
    return HTMLResponse(content=hocr_html, status_code=200, headers=headers)


async def _call_layout_parsing(body: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(LAYOUT_PARSING_URL, json=body)
            try:
                payload = resp.json()
            except Exception:
                payload = {
                    "errorCode": resp.status_code,
                    "errorMsg": "layout-parsing returned non-JSON response",
                    "raw": resp.text,
                }
            return payload, resp.status_code
    except httpx.TimeoutException:
        return {"errorCode": 504, "errorMsg": "layout-parsing timeout"}, 504
    except Exception as exc:
        return {
            "errorCode": 500,
            "errorMsg": "failed to call layout-parsing",
            "detail": str(exc),
        }, 500


def _extract_conversion_source(layout_json: Any) -> Any:
    unwrapped = _unwrap_wrappers(layout_json)
    candidate = _find_convertible_node(unwrapped)
    if candidate is not None:
        return candidate
    return unwrapped


def _unwrap_wrappers(data: Any, max_depth: int = 8) -> Any:
    current = data
    wrapper_keys = ("result", "results", "output", "outputs", "data", "response")

    for _ in range(max_depth):
        parsed = _try_parse_json_string(current)
        if parsed is not None:
            current = parsed
            continue

        if isinstance(current, dict):
            moved = False
            for key in wrapper_keys:
                candidate = current.get(key)
                if candidate is not None:
                    current = candidate
                    moved = True
                    break
            if moved:
                continue

            if len(current) == 1:
                only_val = next(iter(current.values()))
                if only_val is not None:
                    current = only_val
                    continue

        if isinstance(current, list) and len(current) == 1:
            current = current[0]
            continue

        break

    return current


def _find_convertible_node(data: Any) -> Any:
    queue = [data]
    max_nodes = 5000
    visited = 0

    while queue and visited < max_nodes:
        node = queue.pop(0)
        visited += 1

        parsed = _try_parse_json_string(node)
        if parsed is not None:
            queue.append(parsed)
            continue

        if _looks_convertible(node):
            return node

        if isinstance(node, dict):
            preferred_keys = [
                "result",
                "results",
                "output",
                "outputs",
                "data",
                "response",
                "children",
                "blocks",
                "parsing_res_list",
                "spotting_res",
            ]
            for key in preferred_keys:
                if key in node:
                    queue.append(node[key])
            for key, value in node.items():
                if key not in preferred_keys:
                    queue.append(value)
        elif isinstance(node, list):
            queue.extend(node)

    return None


def _looks_convertible(node: Any) -> bool:
    if isinstance(node, dict):
        children = node.get("children")
        if isinstance(children, list) and len(children) > 0:
            return True

        blocks = node.get("blocks")
        if isinstance(blocks, list) and len(blocks) > 0:
            return True

        rec_texts = node.get("rec_texts")
        rec_polys = node.get("rec_polys")
        if isinstance(rec_texts, list) and isinstance(rec_polys, list):
            return True

        parsing_res_list = node.get("parsing_res_list")
        if isinstance(parsing_res_list, list) and len(parsing_res_list) > 0:
            return True

        spotting = node.get("spotting_res")
        if isinstance(spotting, dict):
            s_texts = spotting.get("rec_texts")
            s_polys = spotting.get("rec_polys")
            if isinstance(s_texts, list) and isinstance(s_polys, list):
                return True

    if isinstance(node, list) and node:
        first = node[0]
        if isinstance(first, dict):
            return _looks_convertible(first)

    return False


def _try_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return None

    try:
        return json.loads(text)
    except Exception:
        return None


def _dump_request_json(
    request_log_id: str,
    request_body: Dict[str, Any],
    layout_json: Any,
    conversion_source: Any,
    status_code: int,
) -> None:
    dump_dir = (JSON_DUMP_DIR or "").strip()
    if not dump_dir:
        return

    try:
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_log_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(request_log_id or "no-logid"))
        filename = f"{stamp}_{safe_log_id}_{uuid4().hex[:8]}.json"
        output_path = out_dir / filename

        safe_request_body = _sanitize_for_dump(request_body)
        safe_layout_json = _sanitize_for_dump(layout_json)
        safe_conversion_source = _sanitize_for_dump(conversion_source)

        payload = {
            "timestamp": stamp,
            "requestLogId": request_log_id,
            "statusCode": status_code,
            "requestBody": safe_request_body,
            "layoutJson": safe_layout_json,
            "conversionSource": safe_conversion_source,
            "conversionSourceSummary": _summarize_payload(safe_conversion_source),
        }

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved request JSON to %s", output_path)
    except Exception as exc:
        logger.warning("Failed to dump request JSON: %s", exc)


def _sanitize_for_dump(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, inner in value.items():
            out[str(key)] = _sanitize_for_dump(inner)
        return out

    if isinstance(value, list):
        return [_sanitize_for_dump(item) for item in value]

    if isinstance(value, str):
        if _looks_like_base64_payload(value):
            return f"[omitted-base64 length={len(value)}]"
        return value

    return value


def _looks_like_base64_payload(text: str) -> bool:
    t = text.strip()
    if len(t) < 1024:
        return False

    if t.startswith("data:image/") and ";base64," in t:
        return True

    if re.search(r"https?://", t):
        return False

    compact = re.sub(r"\s+", "", t)
    if len(compact) < 1024:
        return False

    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False

    # JPEG/PNG-like base64 starts are common in OCR inputs.
    if compact.startswith("/9j/") or compact.startswith("iVBOR"):
        return True

    # Heuristic for generic base64 blocks.
    return len(compact) % 4 == 0


def _summarize_payload(payload: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"type": type(payload).__name__}
    if isinstance(payload, dict):
        summary["keys"] = sorted(payload.keys())[:40]
        summary["size"] = len(payload)
    elif isinstance(payload, list):
        summary["size"] = len(payload)
        if payload:
            summary["first_type"] = type(payload[0]).__name__
            if isinstance(payload[0], dict):
                summary["first_keys"] = sorted(payload[0].keys())[:20]
    elif isinstance(payload, str):
        summary["length"] = len(payload)
    return summary


def _extract_image_size(request_body: Dict[str, Any], conversion_source: Any) -> Tuple[int, int]:
    width = _find_dimension_value(request_body, ("width", "image_width", "imageWidth", "w"))
    height = _find_dimension_value(request_body, ("height", "image_height", "imageHeight", "h"))

    if width and height:
        return width, height

    # Try extracting from common tensor shapes in request payload.
    req_shape_w, req_shape_h = _extract_size_from_inputs_shape(request_body)
    width = width or req_shape_w
    height = height or req_shape_h
    if width and height:
        return width, height

    # Try to infer from conversion source geometry.
    src_width, src_height = _extract_size_from_result_geometry(conversion_source)
    width = width or src_width
    height = height or src_height

    return width or 1000, height or 1000


def _find_dimension_value(obj: Any, keys: Tuple[str, ...]) -> Optional[int]:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                parsed = _to_int(obj[key])
                if parsed is not None:
                    return parsed
        for value in obj.values():
            found = _find_dimension_value(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_dimension_value(item, keys)
            if found is not None:
                return found
    return None


def _extract_size_from_inputs_shape(body: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    inputs = body.get("inputs") if isinstance(body, dict) else None
    if not isinstance(inputs, list):
        return None, None

    for item in inputs:
        if not isinstance(item, dict):
            continue
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            continue
        nums = [_to_int(v) for v in shape]
        nums = [n for n in nums if n is not None and n > 0]
        if len(nums) >= 2:
            # Heuristic: for NCHW/NHWC-like tensors take last two dims.
            h = nums[-2]
            w = nums[-1]
            return w, h
    return None, None


def _extract_size_from_result_geometry(result: Any) -> Tuple[Optional[int], Optional[int]]:
    max_x = 0
    max_y = 0

    def walk(obj: Any) -> None:
        nonlocal max_x, max_y
        if isinstance(obj, dict):
            bbox = obj.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                x2 = _to_int(bbox[2])
                y2 = _to_int(bbox[3])
                if x2:
                    max_x = max(max_x, x2)
                if y2:
                    max_y = max(max_y, y2)
            polygon = obj.get("polygon")
            if isinstance(polygon, list):
                for point in polygon:
                    if isinstance(point, list) and len(point) >= 2:
                        x = _to_int(point[0])
                        y = _to_int(point[1])
                        if x:
                            max_x = max(max_x, x)
                        if y:
                            max_y = max(max_y, y)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for elem in obj:
                walk(elem)

    walk(result)
    if max_x > 0 and max_y > 0:
        return max_x, max_y
    return None, None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None
