# services/audio_processor.py

from services.sarvam_service import (
    transcribe_audio,
    analyze_conversation_tags,
    local_tag_analysis
)


def process_audio(audio_path):

    try:

        # =====================================================
        # TRANSCRIBE AUDIO
        # =====================================================
        transcription_result = transcribe_audio(audio_path)

        # =====================================================
        # GET SEGMENTS
        # =====================================================
        raw_segments = transcription_result.get("segments", [])

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

            raw_speaker = str(
                seg.get("speaker", "")
            ).strip()

            # Empty speaker fallback
            if not raw_speaker:
                raw_speaker = f"unknown_{id_counter}"

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