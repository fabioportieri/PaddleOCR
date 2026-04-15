from fastapi import FastAPI, Response, UploadFile, File
import paddlex as pdx
import numpy as np
import cv2
import os
from paddle_hocr_layout import paddlex_to_hocr_layout
from paddlex_adapter import paddlex_to_normalized

app = FastAPI()

DEVICE = os.getenv("DEVICE", "gpu:0")

# -----------------------------
# Pipeline 1: Document Structure
# -----------------------------
doc_pipeline = pdx.create_pipeline(
    pipeline="PP-StructureV3",
    device=DEVICE,
)

# -----------------------------
# Pipeline 2: OCR (text only)
# -----------------------------
# ocr_pipeline = pdx.create_pipeline(
#     pipeline="PP-OCRv5",   # or "PP-OCRv4" depending on your PaddleX version
#     device=DEVICE,
# )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_name = file.filename

    # -----------------------------
    # Run Document Structure
    # -----------------------------
    doc_result_gen = doc_pipeline.predict(
        input=img,
        params={
            "layout_threshold": 0.3,
            "layout_nms": True,
            "text_det_limit_side_len": 960,
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.5,
            "text_rec_score_thresh": 0.5,
            "use_doc_orientation_classify": False,
            "use_table_recognition": True
        }
    )

    # doc_output = [res.json for res in doc_result_gen]

    # -----------------------------
    # Run OCR pipeline
    # -----------------------------
    # ocr_result_gen = ocr_pipeline.predict(
    #     input=img
    # )
    # ocr_output = [res.json for res in ocr_result_gen]

    # Collect all results from generator (consume once)
    doc_results = list(doc_result_gen)

    # -----------------------------
    # Dump doc_output to JSON file for inspection
    # -----------------------------
    import json
    output_dir = "/data/debug"  # Make sure this is a host-mounted volume
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_doc_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        doc_list = [r.json if hasattr(r, "json") else r for r in doc_results]
        json.dump(doc_list, f, ensure_ascii=False, indent=2)

    page = paddlex_to_normalized(doc_results[0])

    # Dump NormalizedPage for inspection
    import dataclasses
    normalized_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_normalized.json")
    with open(normalized_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(page), f, ensure_ascii=False, indent=2)

    hocr = paddlex_to_hocr_layout(page, image_name=image_name)
    return Response(content=hocr, media_type="text/html")