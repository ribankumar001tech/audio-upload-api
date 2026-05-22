# services/audio_processor.py

import os
import re
import shutil
import subprocess

from services.sarvam_service import (
    transcribe_audio,
    analyze_conversation_tags,
    local_tag_analysis
)


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip()).strip()


def _clean_raw_segments(raw_segments):
    """Deduplicate and merge similar adjacent speaker segments."""
    if not raw_segments:
        return []

    sorted_segments = sorted(
        raw_segments,
        key=lambda s: (
            float(s.get("start_time", 0)) if s.get("start_time") is not None else 0,
            float(s.get("end_time", 0)) if s.get("end_time") is not None else 0,
            int(s.get("id", 0)) if s.get("id") is not None else 0,
        ),
    )

    normalized = []
    seen = set()
    for seg in sorted_segments:
        text = _normalize_text(seg.get("text", ""))
        if not text:
            continue

        speaker = str(seg.get("speaker", "") or "").strip()
        key = (speaker, text, seg.get("start_time"), seg.get("end_time"))
        if key in seen:
            continue
        seen.add(key)

        if normalized and speaker == normalized[-1]["speaker"]:
            prev = normalized[-1]
            start_prev = float(prev.get("start_time", 0)) if prev.get("start_time") is not None else 0
            end_prev = float(prev.get("end_time", 0)) if prev.get("end_time") is not None else 0
            start_cur = float(seg.get("start_time", 0)) if seg.get("start_time") is not None else 0
            gap = start_cur - end_prev
            if gap <= 0.75:
                prev["text"] = f"{prev['text']} {text}".strip()
                prev["end_time"] = max(end_prev, float(seg.get("end_time", end_prev)))
                continue

        cleaned_seg = dict(seg)
        cleaned_seg["text"] = text
        normalized.append(cleaned_seg)

    return normalized


def apply_ffmpeg_noise_filter(input_path, output_path):
    """Apply a simple FFmpeg noise reduction filter before transcription."""
    if shutil.which("ffmpeg") is None:
        print("FFmpeg not installed; skipping noise filtering.")
        return input_path

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            "highpass=f=200,afftdn",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"FFmpeg noise filter complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"FFmpeg noise filter failed, falling back to original audio. Error: {e}")
        return input_path


def process_audio(audio_path):

    try:

        # =====================================================
        # AUDIO PREPROCESSING
        # =====================================================
        cleaned_path = os.path.join(os.path.dirname(audio_path), "ffmpeg_filtered.wav")
        audio_to_transcribe = apply_ffmpeg_noise_filter(audio_path, cleaned_path)

        cleanup_paths = set()
        if audio_to_transcribe != audio_path:
            cleanup_paths.add(audio_to_transcribe)

        try:
            transcription_result = transcribe_audio(audio_to_transcribe)
        finally:
            for path in cleanup_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        # =====================================================
        # GET SEGMENTS
        # =====================================================
        raw_segments = transcription_result.get("segments", [])
        raw_segments = _clean_raw_segments(raw_segments)

        if not raw_segments:
            raise Exception(
                f"Empty transcript segments returned. Full response: {transcription_result}"
            )

        # =====================================================
        # DYNAMIC SPEAKER MAPPING
        # =====================================================
        speaker_map = {}
        speaker_counter = 1

        cleaned_segments = []
        formatted_segments = []

        id_counter = 1

        for seg in raw_segments:

            text = str(seg.get("text", "")).strip()

            if not text:
                continue

            raw_speaker = str(seg.get("speaker", "") or "").strip()

            # Empty speaker fallback – alternate unlabeled segments between speaker 1 and 2
            if not raw_speaker:
                raw_speaker = f"unknown_{((id_counter - 1) % 2) + 1}"

            # Dynamic speaker mapping
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker = speaker_map[raw_speaker]

            clean_segment = {
                "call_id": seg.get("call_id", 1),
                "confidence": seg.get("confidence", 1.0),
                "created_at": seg.get("created_at"),
                "end_time": seg.get("end_time"),
                "id": id_counter,
                "language": seg.get("language", "en-IN"),
                "original_text": seg.get("original_text"),
                "speaker": speaker,
                "start_time": seg.get("start_time"),
                "text": text
            }

            cleaned_segments.append(clean_segment)

            # Transcript for AI analysis
            formatted_segments.append(
                f"{speaker}: {text}"
            )

            id_counter += 1

        # =====================================================
        # TRANSCRIPT FOR AI
        # =====================================================
        transcript_text = "\n".join(
            formatted_segments
        ).strip()

        if not transcript_text:
            raise Exception(
                "Transcript text is empty after conversion"
            )

        # =====================================================
        # SARVAM AI TAG ANALYSIS
        # =====================================================
        tags_result = analyze_conversation_tags(
            transcript_text
        )

        # Debugging: show raw tags_result to help identify unexpected shapes
        # try:
        #     print("DEBUG: tags_result ->", tags_result)
        # except Exception:
        #     print("DEBUG: tags_result (unprintable)")

        # =====================================================
        # SAFE TAG FALLBACK
        # =====================================================
        final_tags = {
            "type": [],
            "tone": ["Neutral"],
            "pattern": "Occasional",
            "frequency": "Rare",
            "focus_area": "General",
            "emotional_signal": "Neutral"
        }

        # =====================================================
        # USE AI TAGS IF AVAILABLE
        # =====================================================
        if isinstance(tags_result, dict):

            final_tags["type"] = tags_result.get(
                "type",
                []
            )

            final_tags["tone"] = tags_result.get(
                "tone",
                ["Neutral"]
            )

            final_tags["pattern"] = tags_result.get(
                "pattern",
                "Occasional"
            )

            final_tags["frequency"] = tags_result.get(
                "frequency",
                "Rare"
            )

            final_tags["focus_area"] = tags_result.get(
                "focus_area",
                "General"
            )

            final_tags["emotional_signal"] = tags_result.get(
                "emotional_signal",
                "Neutral"
            )

        # =====================================================
        # Merge local heuristic for critical overrides
        # If local heuristic finds a stronger focus/frequency/pattern, prefer it
        try:
            local = local_tag_analysis(transcript_text)
            if isinstance(local, dict):
                # prefer local focus if it is not General
                if local.get("focus_area") and local.get("focus_area") != "General":
                    final_tags["focus_area"] = local.get("focus_area")

                # prefer local frequency if it's more than Rare
                if local.get("frequency") and local.get("frequency") != "Rare":
                    final_tags["frequency"] = local.get("frequency")

                # prefer local pattern if it indicates escalation or higher
                if local.get("pattern") and local.get("pattern") != "Occasional":
                    final_tags["pattern"] = local.get("pattern")
        except Exception:
            pass

        # =====================================================
        # FINAL RESPONSE
        # =====================================================
        return {
            "success": True,
            "job_id": transcription_result.get("job_id"),
            "status": "completed",
            "tags": final_tags,
            "transcript": cleaned_segments
        }

    except Exception as e:

        print(str(e))

        return {
            "success": False,
            "error": str(e)
        }