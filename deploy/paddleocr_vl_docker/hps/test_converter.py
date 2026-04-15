import json
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


# --- CONFIG ---
JSON_FILE = "input.json"   # your JSON file
WIDTH = 1216
HEIGHT = 1500
LANG = "eng"


def load_converter():
    converter_path = Path("sources/paddle_hocr_converter2.py").resolve()

    spec = spec_from_file_location("paddle_hocr_converter2_dynamic", converter_path)
    module = module_from_spec(spec)

    # CRITICAL: fix dataclass error
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module.PaddleHOCRConverter2


def extract_conversion_source(data):
    """
    Minimal version of your API logic:
    tries to extract the real payload used by converter
    """
    try:
        return data["layoutJson"]["result"]["layoutParsingResults"][0]["prunedResult"]
    except Exception:
        return data  # fallback


def main():
    # load json
    data = json.loads(Path(JSON_FILE).read_text(encoding="utf-8"))

    # extract correct node
    src = extract_conversion_source(data)

    # load converter
    Converter = load_converter()
    converter = Converter()

    # run conversion
    hocr, text = converter.convert(
        result=src,
        image_width=WIDTH,
        image_height=HEIGHT,
        lang=LANG
    )

    # save outputs
    Path("output.hocr.html").write_text(hocr, encoding="utf-8")
    Path("output.txt").write_text(text, encoding="utf-8")

    print(" Done")
    print("carea count:", hocr.count('ocr_carea'))
    print("text length:", len(text))


if __name__ == "__main__":
    main()
