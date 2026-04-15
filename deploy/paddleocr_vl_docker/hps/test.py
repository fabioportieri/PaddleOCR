import json
import pathlib

# Direct import (now that filename is valid)
from sources.paddle_hocr_converter2 import PaddleHOCRConverter2


def load_input(json_path):
    data = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    return data["layoutJson"]["result"]["layoutParsingResults"][0]["prunedResult"]


def run_test(json_path, width=1216, height=1500, lang="eng"):
    src = load_input(json_path)

    conv = PaddleHOCRConverter2()
    hocr, text = conv.convert(src, width, height, lang)

    print("=== RESULTS ===")
    print("carea count :", hocr.count("ocr_carea"))
    print("text length :", len(text))

    return hocr, text


if __name__ == "__main__":
    # 6363 change filename if needed
    input_file = "20260414T131041.125391Z_no-logid_a822cf8e.json"

    run_test(input_file)
