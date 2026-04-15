from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
import cv2
import base64

app = FastAPI()

ocr = PaddleOCR(
    det_model_dir="/models/ppocrv4/det",
    rec_model_dir="/models/ppocrv4/rec",
    cls_model_dir="/models/ppocrv4/cls",
    use_angle_cls=True,
    lang="en"
)

class OCRRequest(BaseModel):
    image_base64: str


def decode_image(b64):
    img_data = base64.b64decode(b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.post("/ocr")
def ocr_endpoint(req: OCRRequest):
    img = decode_image(req.image_base64)

    result = ocr.ocr(img, cls=True)

    words = []
    for line in result:
        for box, (text, score) in line:
            words.append({
                "text": text,
                "confidence": float(score),
                "bbox": box
            })

    return {"words": words}