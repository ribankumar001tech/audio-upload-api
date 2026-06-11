import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import requests
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()


def _split_audio_chunks(input_path, chunk_len=25):
    """Split `input_path` into chunk_len-second WAV files using ffmpeg.

    Returns a list of file paths or None if splitting isn't possible.
    """
    if shutil.which("ffmpeg") is None:
        return None

    tmpdir = tempfile.mkdtemp(prefix="sarvam_chunks_")
    out_pattern = os.path.join(tmpdir, "chunk_%03d.wav")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_len),
            out_pattern,
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # collect generated chunks
        files = sorted([str(p) for p in Path(tmpdir).glob("chunk_*.wav")])
        return files
    except Exception:
        # cleanup on failure
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return None


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# =====================================================
# TRANSCRIBE AUDIO
# =====================================================

def _normalize_sarvam_diarized_entry(entry, idx, default_language="en-IN"):
    return {
        "call_id": 1,
        "confidence": 1.0,
        "created_at": None,
        "end_time": float(entry.get("end_time_seconds", entry.get("end_time", 0))),
        "id": idx,
        "language": default_language,
        "original_text": entry.get("transcript") or entry.get("text") or None,
        "speaker": str(entry.get("speaker_id") or entry.get("speaker") or "Speaker 1").strip(),
        "start_time": float(entry.get("start_time_seconds", entry.get("start_time", 0))),
        "text": str(entry.get("transcript") or entry.get("text") or "").strip(),
    }


def _parse_sarvam_batch_output(data):
    diarized = data.get("diarized_transcript")
    if isinstance(diarized, dict):
        entries = diarized.get("entries") or []
    else:
        entries = []

    if not entries:
        return None

    default_language = data.get("language_code", "en-IN") or "en-IN"
    segments = [
        _normalize_sarvam_diarized_entry(entry, idx + 1, default_language=default_language)
        for idx, entry in enumerate(entries)
        if entry.get("transcript")
    ]

    if not segments:
        return None

    return {
        "success": True,
        "segments": segments,
        "raw": data,
    }


def _transcribe_audio_batch(audio_path):
    if not SARVAM_API_KEY:
        return {"success": False, "error": "SARVAM_API_KEY is not configured"}

    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        job = client.speech_to_text_job.create_job(
            model="saarika:v2.5",
            language_code="unknown",
            with_diarization=True,
            with_timestamps=True,
        )

        job.upload_files([audio_path])
        job.start()
        job.wait_until_complete(timeout=600)

        temp_dir = tempfile.mkdtemp(prefix="sarvam_batch_output_")
        try:
            job.download_outputs(temp_dir)
            json_files = glob.glob(os.path.join(temp_dir, "*.json"))
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                result = _parse_sarvam_batch_output(data)
                if result:
                    return result

            return {"success": False, "error": "No diarized transcript found in Sarvam batch output"}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return {"success": False, "error": f"Sarvam batch diarization failed: {str(e)}"}


def transcribe_audio(audio_path):
    batch_result = _transcribe_audio_batch(audio_path)
    if batch_result.get("success"):
        return batch_result

    # Fallback to the REST endpoint if batch diarization fails
    url = "https://api.sarvam.ai/speech-to-text"

    mime_type, _ = mimetypes.guess_type(audio_path)

    if mime_type is None:
        mime_type = "audio/mpeg"

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    def _upload_file(p):
        try:
            with open(p, "rb") as audio_file:
                files = {"file": (os.path.basename(p), audio_file, mime_type)}
                data = {"model": "saarika:v2.5"}
                return requests.post(url, headers=headers, files=files, data=data, timeout=120)
        except Exception:
            return None

    response = _upload_file(audio_path)
    if response is None:
        return {"success": False, "error": "Failed to open audio file"}

    if response.status_code != 200:
        # handle duration-exceeded error by chunking the audio and uploading pieces
        try:
            body = response.text or ""
            if "duration exceeds the maximum limit" in body.lower() or "exceeds the maximum limit of 30 seconds" in body.lower():
                # prefer to split audio into smaller chunks (25s) and upload sequentially
                # if ffmpeg is not available, return an actionable error message
                if shutil.which("ffmpeg") is None:
                    return {
                        "success": False,
                        "error": (
                            "Audio duration exceeds 30 seconds and ffmpeg is not installed. "
                            "Install ffmpeg (eg. `sudo apt install ffmpeg`) or use Sarvam's batch API for longer files."
                        )
                    }

                chunks = _split_audio_chunks(audio_path, chunk_len=25)
                if not chunks:
                    return {"success": False, "error": response.text}

                combined_segments = []
                seg_id = 1
                for idx, chunk_path in enumerate(chunks):
                    r = _upload_file(chunk_path)
                    if not r or r.status_code != 200:
                        for c in chunks:
                            try:
                                os.remove(c)
                            except Exception:
                                pass
                        return {"success": False, "error": f"Chunk upload failed: {r.text if r is not None else 'no response'}"}

                    data = r.json()
                    text = (
                        data.get("transcript")
                        or data.get("text")
                        or data.get("raw", {}).get("transcript")
                        or ""
                    )
                    lines = [l.strip() for l in text.split("\n") if l.strip()]
                    for i, line in enumerate(lines, start=1):
                        if "," in line:
                            speaker, msg = line.split(",", 1)
                        elif ":" in line:
                            speaker, msg = line.split(":", 1)
                        else:
                            speaker = f"Speaker {((i % 2) + 1)}"
                            msg = line

                        combined_segments.append({
                            "call_id": 1,
                            "confidence": 1.0,
                            "created_at": data.get("created_at", "2026-03-31 12:59:24"),
                            "end_time": idx * 25 + i * 5,
                            "id": seg_id,
                            "language": data.get("language_code", "en-IN"),
                            "original_text": None,
                            "speaker": speaker.strip(),
                            "start_time": idx * 25 + (i - 1) * 5,
                            "text": msg.strip()
                        })
                        seg_id += 1

                for c in chunks:
                    try:
                        os.remove(c)
                    except Exception:
                        pass

                return {"success": True, "segments": combined_segments, "raw": {"chunks": len(chunks)}}
        except Exception:
            return {"success": False, "error": response.text}
        return {"success": False, "error": response.text}

    try:
        data = response.json()
        text = (
            data.get("transcript")
            or data.get("text")
            or data.get("raw", {}).get("transcript")
            or ""
        )

        if not text:
            return {"success": False, "error": "No transcript found", "raw": data}

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        segments = []

        for i, line in enumerate(lines, start=1):
            if "," in line:
                speaker, msg = line.split(",", 1)
            elif ":" in line:
                speaker, msg = line.split(":", 1)
            else:
                speaker = f"Speaker {(i % 2) + 1}"
                msg = line

            segments.append({
                "call_id": 1,
                "confidence": 1.0,
                "created_at": "2026-03-31 12:59:24",
                "end_time": i * 5,
                "id": i,
                "language": data.get("language_code", "en-IN"),
                "original_text": None,
                "speaker": speaker.strip(),
                "start_time": (i - 1) * 5,
                "text": msg.strip()
            })

        return {"success": True, "segments": segments, "raw": data}
    except Exception as e:
        return {"success": False, "error": f"Parse error: {str(e)}"}


def build_transcript_text(segments):
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "Speaker")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# =====================================================
# AI TAG ANALYSIS (PURE SARVAM AI – NO MANUAL FALLBACK)
# =====================================================

def analyze_conversation_tags(transcript_text):
    url = "https://api.sarvam.ai/v1/chat/completions"

    # NOTE: Try "api-subscription-key" if Bearer auth returns 401
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    prompt = f"""You are a conversation analyst specializing in detecting toxic, abusive, and emotionally charged language across English, Hindi, and Hinglish.

Analyze the following conversation transcript and return a JSON object with EXACTLY these keys:

- "type": list — one or more of: ["Harassment", "Insult", "Manipulation", "Intimidation", "Mockery"]
- "tone": list — one or more of: ["Hostile", "Aggressive", "Sarcastic", "Dismissive", "Threatening"]
- "pattern": string — one of: "Isolated", "Occasional", "Frequent", "Escalating", "Cyclical"
- "focus_area": string — one of: "Physical", "Mental", "Academic", "Personal", "Offensive"
- "emotional_signal": string — one of: "Nervous", "Defensive", "Quiet", "Doubtful", "Dazed"

Detection guidelines:
- Detect insults, threats, harassment, mockery, intimidation, or manipulation in ANY language (English, Hindi, Hinglish, regional slang)
- Hindi/Hinglish abuse examples: गधा, बेवकूफ, हरामी, साला, bhad me ja, teri maa, chutiya, etc.
- Workplace intimidation: threats about job, dismissive attitude toward personal life, forced overtime, warnings
- If none of the options apply (e.g., the conversation is calm, professional, or normal), return an empty list [] for "type" and "tone", and an empty string "" for "pattern", "focus_area", and "emotional_signal".
- Escalating pattern: hostility increases over the conversation

Transcript:
\"\"\"
{transcript_text}
\"\"\"

Return ONLY a valid JSON object. No explanation, no markdown, no code fences, no extra text.

Example output:
{{"type": ["Insult", "Harassment"], "tone": ["Hostile", "Aggressive"], "pattern": "Escalating", "focus_area": "Personal", "emotional_signal": "Defensive"}}"""

    payload = {
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    default_result = {
        "type": [],
        "tone": [],
        "pattern": "",
        "frequency": "",
        "focus_area": "",
        "emotional_signal": ""
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        # print(f"[DEBUG] Status: {response.status_code}")
        # print(f"[DEBUG] Response text: {response.text[:500]}")

        if response.status_code != 200:
            # print("[DEBUG] API returned non-200, returning default")
            return default_result

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content:
            # print("[DEBUG] Empty content from API")
            return default_result

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        # print(f"[DEBUG] Parsed AI response: {parsed}")

        # Safely build result with fallbacks for each key
        result = {
            "type": parsed.get("type", []),
            "tone": parsed.get("tone", []),
            "pattern": parsed.get("pattern", ""),
            "frequency": parsed.get("frequency", ""),
            "focus_area": parsed.get("focus_area", ""),
            "emotional_signal": parsed.get("emotional_signal", "")
        }

        # Ensure type and tone are always lists
        if isinstance(result["type"], str):
            result["type"] = [result["type"]]
        if isinstance(result["tone"], str):
            result["tone"] = [result["tone"]]

        # Deduplicate
        result["type"] = list(dict.fromkeys(result["type"]))
        result["tone"] = list(dict.fromkeys(result["tone"]))

        return result

    except json.JSONDecodeError as e:
        # print(f"[DEBUG] JSON parse error: {e} | Raw content: {content}")
        return default_result
    except Exception as e:
        # print(f"[DEBUG] Exception: {e}")
        return default_result


# =====================================================
# LOCAL TAG ANALYSIS (KEPT FOR BACKWARD COMPATIBILITY)
# =====================================================
# This function is NOT called from analyze_conversation_tags anymore.
# It is only kept so that other modules (e.g., audio_processor.py)
# that import it do not break. You can safely remove it if no external
# code depends on it.

def local_tag_analysis(transcript_text: str) -> dict:
    """Simple heuristic-based fallback analysis (DEPRECATED – not used by analyze_conversation_tags)."""
    text = (transcript_text or "").lower()

    insults = ["idiot", "stupid", "dumb", "shit", "bastard", "asshole", "moron"]
    harassment = ["shut up", "get lost", "leave me", "die"]
    threats_verbs = ["kill", "hurt", "destroy", "attack", "beat"]
    sarcasm_cues = ["yeah right", "as if", "sure", "whatever"]
    mockery_cues = ["mock", "mockery", "ridicule", "ridiculous"]

    hindi_insults = ["गधा", "बेवकूफ", "मुर्ख", "चूतिया", "हरामी", "साला", "भाड़ में जाए", "भाड़", "तेरा", "तुम्हारा"]
    hindi_harassment = ["चुप रहो", "जाओ", "निकल", "बंद करो", "जाऊं"]
    hinglish_insults = ["bhad", "bhad me ja", "bhad me jaa", "teri family", "mera kya aata hai", "mera kya"]
    workplace_intimidation = [
        "i don't care about your family",
        "holiday",
        "holidays",
        "you must work",
        "next time onwards",
        "will not participate",
        "recovery section",
        "i dont care",
        "don't care about your family",
        "family",
        "bank",
        "work pressure",
        "warning",
        "participate including holidays",
        "डोंट केयर",
        "फैमिली",
        "छुट्टियां",
        "बैंक",
        "काम कर रहे हैं",
    ]
    hindi_frustration = [
        "पता नहीं",
        "मेरी गलती",
        "मैं क्या करूं",
        "क्या करूं",
        "समझ नहीं",
        "परेशान",
        "नाराज़",
        "गुस्सा",
        "चिढ़",
        "तंग आ गया",
        "घबर",
        "फ्रस्ट",
        "टेंशन",
        "निराश",
        "निराष",
        "बढ़ता जा रहा है",
        "हर कॉल के साथ",
        "मजाक उड़ाना",
        "व्यक्तिगत",
        "व्यक्तिगत लगने लगा",
    ]

    types = []
    tone = []
    emotional = ""
    pattern = ""
    frequency = ""

    if any(w in text for w in insults) or any(w in text for w in hindi_insults) or any(w in text for w in hinglish_insults):
        types.append("Insult")
    if any(w in text for w in harassment) or any(w in text for w in hindi_harassment) or any(w in text for w in hinglish_insults):
        types.append("Harassment")
    if any(w in text for w in threats_verbs) or re.search(r"\b(i will|i'll|i'm going to)\s+(kill|hurt|destroy|attack|beat)\b", text):
        types.append("Intimidation")
    if any(w in text for w in mockery_cues):
        types.append("Mockery")
    if any(w in text for w in workplace_intimidation):
        types.extend(["Intimidation", "Harassment"])
        tone = ["Aggressive", "Dismissive", "Hostile"]
        emotional = "Frustrated"
        pattern = "Escalating"
        frequency = "Frequent"
    if not types:
        types = []

    if any(w in text for w in threats_verbs) or re.search(r"\b(kill|hurt|destroy|attack|beat)\b", text):
        tone = ["Threatening"]
        emotional = "Frustrated"
    elif any(w in text for w in insults) or any(w in text for w in hindi_insults) or any(w in text for w in hinglish_insults):
        tone = ["Hostile"]
        emotional = "Frustrated"
    elif any(w in text for w in sarcasm_cues):
        tone = ["Sarcastic"]
        emotional = "Defensive"
    elif any(w in text for w in mockery_cues):
        tone = ["Hostile"]
        emotional = "Frustrated"
    elif any(w in text for w in hindi_frustration):
        tone = []
        emotional = "Frustrated"
    elif re.search(r"frustrat", text):
        tone = ["Hostile"]
        emotional = "Frustrated"
    elif re.search(r"\?{1,}$", transcript_text.strip()):
        tone = []
        emotional = "Doubtful"
    else:
        tone = []
        emotional = ""

    neg_count = sum(text.count(w) for w in insults + harassment + threats_verbs)
    total_words = max(1, len(re.findall(r"\w+", text)))
    ratio = neg_count / total_words

    if ratio > 0.05:
        frequency = "Frequent"
        pattern = "Frequent"
    elif ratio > 0.01:
        frequency = "Occasional"
        pattern = "Occasional"
    else:
        frequency = ""
        pattern = ""

    if re.search(r"\boccasional\b", text):
        frequency = "Occasional"
    if re.search(r"escalat", text):
        pattern = "Escalating"

    if re.search(r"\b(student|exam|school|college|homework|study)\b", text):
        focus = "Academic"
    elif re.search(r"\b(health|medicine|therapy|mental|depress)\b", text):
        focus = "Mental"
    elif re.search(r"\b(body|hit|punch|kick|beat)\b", text):
        focus = "Physical"
    elif any(w in text for w in workplace_intimidation):
        focus = "Mental"
    elif any(w in text for w in insults + harassment + threats_verbs):
        focus = "Personal"
    else:
        focus = ""

    if re.search(r"\bpersonal\b", text):
        focus = "Personal"

    if re.search(r"भाड़|भाड़ में|bhad\b|bhad me|tera .* kya aata|mera kya aata", text):
        if "Insult" not in types:
            types.append("Insult")
        tone = ["Hostile"]
        emotional = "Frustrated"

    if isinstance(types, list):
        seen = []
        for t in types:
            if t not in seen:
                seen.append(t)
        types = seen

    return {
        "type": types,
        "tone": tone,
        "pattern": pattern,
        "frequency": frequency,
        "focus_area": focus,
        "emotional_signal": emotional
    }