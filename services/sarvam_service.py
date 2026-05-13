# services/sarvam_service.py

import requests
import os
import base64
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"


def transcribe_audio(audio_path):

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "file": audio_b64,
        "language_code": "hi-IN",
        "model": "saarika:v2.5",
        "with_diarization": True
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    response = requests.post(
        SARVAM_API_URL,
        json=payload,   # ✅ THIS IS CRITICAL
        headers=headers
    )

    if response.status_code != 200:
        raise Exception(f"Sarvam API Error: {response.text}")

    return response.json()