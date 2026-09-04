"""OCR for uploaded reports/prescriptions, via local Tesseract.

Not a hosted vision model -- this Groq account has none available (model
list is text + Whisper only). Tesseract is only loaded while processing an
upload, so it's cheap to keep alongside everything else on the t3.small.

If the tesseract binary isn't installed, everything here degrades to a
clear message instead of crashing the app.
"""

import io

from src.config import LOGGER

MIN_USEFUL_CHARS = 25  # below this it's noise from a blurry photo, not text
MAX_OCR_CHARS = 6_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class OCRUnavailable(RuntimeError):
    """Tesseract binary isn't installed."""


def ocr_available() -> bool:
    try:
        import pytesseract
        from PIL import Image  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image(image_bytes: bytes) -> str:
    """Just recovers text -- no medical interpretation happens here."""
    if not image_bytes:
        return ""
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailable(
            "OCR requires the 'pytesseract' and 'Pillow' packages."
        ) from exc

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("L")  # greyscale helps on phone photos
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        if "tesseract" in str(exc).lower():
            raise OCRUnavailable(
                "The Tesseract OCR engine is not installed on this server."
            ) from exc
        LOGGER.warning("OCR failed for uploaded image", exc_info=True)
        raise ValueError("Could not read this image. Try a clearer photo.") from exc

    return clean_ocr_text(text)


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # drop punctuation-only lines (table borders, scan edges)
        if stripped and any(char.isalnum() for char in stripped):
            lines.append(stripped)

    cleaned = "\n".join(lines).strip()
    return cleaned[:MAX_OCR_CHARS]


def is_useful(text: str) -> bool:
    return len(text.strip()) >= MIN_USEFUL_CHARS
