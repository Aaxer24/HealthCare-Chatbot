"""OCR for uploaded reports and prescriptions.

Uses Tesseract locally rather than a hosted vision model: this Groq account
exposes no vision-capable model (the model list is text + Whisper only), so
there is no API route for image understanding.

Tesseract is only resident while an upload is being processed, which is what
makes it affordable on a 2 GB t3.small alongside PyTorch, the embedding model,
the reranker, FAISS and the BM25 index.

The binary is an OS package, not a Python one -- if it is missing, every
function here degrades to a clear message rather than raising, so the chatbot
keeps working without the upload feature.
"""

import io

from src.config import LOGGER

# Anything shorter is almost certainly noise from a blurry photo rather than a
# real document, and feeding it to the model would just invite hallucination.
MIN_USEFUL_CHARS = 25
MAX_OCR_CHARS = 6_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class OCRUnavailable(RuntimeError):
    """Raised when the Tesseract binary is not installed."""


def ocr_available() -> bool:
    try:
        import pytesseract
        from PIL import Image  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image(image_bytes: bytes) -> str:
    """Return text found in an image. Raises OCRUnavailable if Tesseract is missing.

    Deliberately does no medical interpretation -- it only recovers text, which
    is then passed to the normal RAG pipeline as context.
    """
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
        # Greyscale helps Tesseract on phone photos of printed reports and
        # costs nothing compared with the OCR pass itself.
        image = image.convert("L")
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        # pytesseract raises TesseractNotFoundError when the binary is absent;
        # catching broadly keeps a corrupt upload from looking like a crash.
        if "tesseract" in str(exc).lower():
            raise OCRUnavailable(
                "The Tesseract OCR engine is not installed on this server."
            ) from exc
        LOGGER.warning("OCR failed for uploaded image", exc_info=True)
        raise ValueError("Could not read this image. Try a clearer photo.") from exc

    return clean_ocr_text(text)


def clean_ocr_text(text: str) -> str:
    """Tidy raw OCR output into something worth sending to the model."""
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop lines that are pure punctuation noise, a common OCR artefact on
        # table borders and scan edges.
        if stripped and any(char.isalnum() for char in stripped):
            lines.append(stripped)

    cleaned = "\n".join(lines).strip()
    return cleaned[:MAX_OCR_CHARS]


def is_useful(text: str) -> bool:
    return len(text.strip()) >= MIN_USEFUL_CHARS
