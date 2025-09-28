# routes_media.py
import io
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import pytesseract

router = APIRouter(prefix="/media", tags=["media"])

# Optional: allow specifying tesseract binary path via env on Windows
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class TranscriptionOut(BaseModel):
    text: str


@router.post("/image/ocr", response_model=TranscriptionOut)
async def image_ocr(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        return TranscriptionOut(text=text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# Voice transcription with Vosk (local, free). Requires model directory.
try:
    import vosk  # type: ignore
    import wave
except Exception:
    vosk = None  # type: ignore

VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR")  # e.g., ./vosk-model-small-en-us-0.15
_vosk_model = None


def _get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        if not VOSK_MODEL_DIR or not Path(VOSK_MODEL_DIR).exists():
            raise RuntimeError("VOSK_MODEL_DIR is not configured or not found")
        _vosk_model = vosk.Model(VOSK_MODEL_DIR)
    return _vosk_model


@router.post("/audio/transcribe", response_model=TranscriptionOut)
async def audio_transcribe(audio: UploadFile = File(...)):
    if vosk is None:
        raise HTTPException(status_code=500, detail="Vosk not installed. Install 'vosk' and set VOSK_MODEL_DIR.")

    try:
        # Expect WAV (16k mono). If other format is sent, frontend should convert or backend can be extended.
        data = await audio.read()
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise HTTPException(status_code=400, detail="Send 16-bit mono PCM WAV for best results")
            rec = vosk.KaldiRecognizer(_get_vosk_model(), wf.getframerate())
            text = ""
            while True:
                buf = wf.readframes(4000)
                if len(buf) == 0:
                    break
                if rec.AcceptWaveform(buf):
                    pass
            # final result
            res = rec.FinalResult()
            # res is a JSON string like {"text": "..."}
            import json
            text = json.loads(res).get("text", "")
            return TranscriptionOut(text=text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
