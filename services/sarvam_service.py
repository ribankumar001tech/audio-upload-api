# services/sarvam_service.py

import os
import json
import requests
import mimetypes

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# =====================================================
# TRANSCRIBE AUDIO
# =====================================================

def transcribe_audio(audio_path):

    url = "https://api.sarvam.ai/speech-to-text"

    mime_type, _ = mimetypes.guess_type(audio_path)

    if mime_type is None:
        mime_type = "audio/mpeg"

    # print("Detected MIME TYPE:", mime_type)

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

        lines = [
            l.strip()
            for l in text.split("\n")
            if l.strip()
        ]

        segments = []

        for i, line in enumerate(lines, start=1):

            if "," in line:

                speaker, msg = line.split(",", 1)

            elif ":" in line:

                speaker, msg = line.split(":", 1)

            else:

                # Auto alternating speakers
                speaker = f"Speaker {(i % 2) + 1}"
                msg = line

            segments.append({

                "call_id": 1,
                "confidence": 1.0,
                "created_at": "2026-03-31 12:59:24",
                "end_time": i * 5,
                "id": i,
                "language": data.get(
                    "language_code",
                    "en-IN"
                ),
                "original_text": None,
                "speaker": speaker.strip(),
                "start_time": (i - 1) * 5,
                "text": msg.strip()

            })

        return {

            "success": True,
            "segments": segments,
            "raw": data

        }

    except Exception as e:

        return {
            "success": False,
            "error": f"Parse error: {str(e)}"
        }


# =====================================================
# AI TAG ANALYSIS
# =====================================================

def analyze_conversation_tags(transcript_text):

    url = "https://api.sarvam.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Analyze the following conversation transcript.

Return ONLY valid JSON.

JSON format:

{{
    "type": [],
    "tone": [],
    "pattern": "",
    "frequency": "",
    "focus_area": "",
    "emotional_signal": ""
}}

Rules:

- type:
Harassment,
Insult,
Manipulation,
Intimidation,
Mockery

- tone:
Hostile,
Aggressive,
Sarcastic,
Dismissive,
Threatening,
Polite,
Neutral

- pattern:
Isolated,
Occasional,
Frequent,
Escalating,
Cyclical

- frequency:
Prevalent,
Frequent,
Occasional,
Rare,
Never

- focus_area:
Physical,
Mental,
Academic,
Personal,
Offensive,
General

- emotional_signal:
Nervous,
Defensive,
Quiet,
Doubtful,
Dazed,
Satisfied,
Frustrated,
Neutral

IMPORTANT:
- Always fill every field
- Never return empty JSON
- If conversation is normal:
  tone=["Neutral"]
  emotional_signal="Neutral"

Transcript:
{transcript_text}
"""

    payload = {

        "model": "sarvam-m",

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.2

    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:

            return {
                "type": [],
                "tone": ["Neutral"],
                "pattern": "Occasional",
                "frequency": "Rare",
                "focus_area": "General",
                "emotional_signal": "Neutral"
            }

        data = response.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )

        # Clean markdown if returned
        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )


        parsed = json.loads(content)

        # If type is empty list, set to ["Normal"]
        type_value = parsed.get("type", ["Normal"])
        if not type_value:
            type_value = ["Normal"]

        return {
            "type": type_value,
            "tone": parsed.get(
                "tone",
                ["Neutral"]
            ),
            "pattern": parsed.get(
                "pattern",
                "Occasional"
            ),
            "frequency": parsed.get(
                "frequency",
                "Rare"
            ),
            "focus_area": parsed.get(
                "focus_area",
                "General"
            ),
            "emotional_signal": parsed.get(
                "emotional_signal",
                "Neutral"
            )
        }

    except Exception as e:

        print(str(e))

        return {
            "type": ["Neutral"],
            "tone": ["Neutral"],
            "pattern": "Occasional",
            "frequency": "Rare",
            "focus_area": "General",
            "emotional_signal": "Neutral"
        }