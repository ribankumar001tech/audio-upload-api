# services/audio_processor.py

from services.sarvam_service import transcribe_audio
from services.context_ollama import analyze_transcription


def process_audio(audio_path):

    try:

        # =====================================================
        # TRANSCRIBE AUDIO
        # =====================================================
        transcription_result = transcribe_audio(audio_path)

        print("\n====== TRANSCRIPTION RESULT ======\n")
        print(transcription_result)
        print("\n==================================\n")

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

            # Transcript for AI
            formatted_segments.append(
                f"{speaker}: {text}"
            )

            id_counter += 1

        # =====================================================
        # TRANSCRIPT FOR AI
        # =====================================================
        transcript_text = "\n".join(formatted_segments).strip()

        if not transcript_text:
            raise Exception(
                "Transcript text is empty after conversion"
            )

        print("\n====== FORMATTED TRANSCRIPT ======\n")
        print(transcript_text)
        print("\n==================================\n")

        # =====================================================
        # AI ANALYSIS
        # =====================================================
        analysis_result = analyze_transcription(
            transcript_text
        )

        print("\n====== ANALYSIS RESULT ======\n")
        print(analysis_result)
        print("\n=============================\n")

        # =====================================================
        # SAFE FALLBACK
        # =====================================================
        if not analysis_result:
            analysis_result = {}

        final_tags = analysis_result.get("tags", {})

        # =====================================================
        # DEFAULT TAGS IF AI FAILS
        # =====================================================
        if not final_tags:

            final_tags = {
                "type": [],
                "tone": ["Neutral"],
                "pattern": "Occasional",
                "frequency": "Rare",
                "focus_area": "General",
                "emotional_signal": "Neutral"
            }

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

        print("\n====== PROCESS ERROR ======\n")
        print(str(e))
        print("\n===========================\n")

        return {
            "success": False,
            "error": str(e)
        }