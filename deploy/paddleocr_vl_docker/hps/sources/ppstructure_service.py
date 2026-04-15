from fastapi import FastAPI, UploadFile, File
from paddleocr import PPStructureV3
import numpy as np
import cv2
import os

app = FastAPI()

# init once (important for performance)
pipeline = PPStructureV3(
  # Layout tuning
    layout_merge_bboxes_mode="large",
    layout_threshold=0.3,
    layout_nms=True,
    layout_unclip_ratio=1.5,

    # Table (CRITICAL)
    use_table_recognition=True,
    use_region_detection=True,

    # OCR tuning
    text_det_thresh=0.2,
    text_det_box_thresh=0.4,
    text_rec_score_thresh=0.5,

    # Disable extras
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    use_seal_recognition=False,
    use_formula_recognition=False,
    use_chart_recognition=False,

    # Output
    format_block_content=False,
    markdown_ignore_labels=[],
    device=os.getenv("DEVICE", "cpu")
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # decode image
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = pipeline.predict(img)

    output = []
    for res in results:
        output.append(res.to_dict())  # structured JSON

    return {
        "status": "success",
        "results": output
    }