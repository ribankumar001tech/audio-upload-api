# services/sarvam_service.py

import requests
import mimetypes
import os

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


def transcribe_audio(audio_path):

    url = "https://api.sarvam.ai/speech-to-text"

    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/mpeg"

    print("Detected MIME TYPE:", mime_type)

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    try:
        with open(audio_path, "rb") as audio_file:

            files = {
                "file": (
                    os.path.basename(audio_path),
                    audio_file,
                    mime_type
                )
            }

            data = {
                "model": "saarika:v2.5"
            }

            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    print("\n====== SARVAM RESPONSE ======\n")
    print(response.status_code)
    print(response.text)
    print("\n=============================\n")

    if response.status_code != 200:
        return {
            "success": False,
            "error": response.text
        }

    try:
        data = response.json()

        text = (
            data.get("transcript")
            or data.get("text")
            or data.get("raw", {}).get("transcript")
            or ""
        )

        if not text:
            return {
                "success": False,
                "error": "No transcript found",
                "raw": data
            }

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        segments = []

        for i, line in enumerate(lines, start=1):

            if "," in line:
                speaker, msg = line.split(",", 1)
            else:
                speaker, msg = "Unknown", line

            segments.append({
                "call_id": 1,
                "confidence": 1.0,
                "created_at": "2026-03-31 12:59:24",
                "end_time": i * 5,
                "id": i,
                "language": data.get("language_code", "hi"),
                "original_text": None,
                "speaker": speaker.strip(),
                "start_time": (i - 1) * 5,
                "text": msg.strip()
            })

        return {
            "success": True,
            "segments": segments,
            "tags": data.get("tags", {}),
            "raw": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Parse error: {str(e)}"
        }