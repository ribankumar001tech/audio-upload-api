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
        # GET STRUCTURED TRANSCRIPT
        # =====================================================
        raw_segments = transcription_result.get("segments", [])

        if not raw_segments:
            raise Exception(
                f"Empty transcript segments returned. Full response: {transcription_result}"
            )

        # =====================================================
        # NORMALIZE SPEAKERS
        # =====================================================
        def normalize_speaker(speaker, last_speaker=None):

            if not speaker:
                speaker = ""

            speaker = str(speaker).strip().lower()

            # Sarvam generic speakers
            if speaker in ["speaker 1", "speaker1"]:
                return "Agent"

            if speaker in ["speaker 2", "speaker2"]:
                return "Customer"

            # Already correct
            if speaker == "agent":
                return "Agent"

            if speaker == "customer":
                return "Customer"

            # Unknown / empty fallback
            if speaker in ["", "unknown", "none", "null"]:
                if last_speaker == "Agent":
                    return "Customer"
                return "Agent"

            # Keep original if custom speaker exists
            return speaker.title()

        # =====================================================
        # BUILD CLEAN TRANSCRIPT
        # =====================================================
        cleaned_segments = []
        formatted_segments = []

        last_speaker = None
        id_counter = 1

        for seg in raw_segments:

            text = str(seg.get("text", "")).strip()

            if not text:
                continue

            speaker = normalize_speaker(
                seg.get("speaker"),
                last_speaker
            )

            last_speaker = speaker

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

            formatted_segments.append(
                f"{speaker}: {text}"
            )

            id_counter += 1

        # =====================================================
        # TRANSCRIPT FOR LLM
        # =====================================================
        transcript_text = "\n".join(formatted_segments).strip()

        if not transcript_text:
            raise Exception("Transcript text is empty after conversion")

        print("\n====== FORMATTED TRANSCRIPT ======\n")
        print(transcript_text)
        print("\n==================================\n")

        # =====================================================
        # ANALYZE TRANSCRIPT
        # =====================================================
        analysis_result = analyze_transcription(transcript_text)

        print("\n====== ANALYSIS RESULT ======\n")
        print(analysis_result)
        print("\n=============================\n")

        if not analysis_result:
            analysis_result = {}

        # =====================================================
        # RETURN FINAL RESPONSE
        # =====================================================
        return {
            "success": True,
            "transcript": cleaned_segments,
            "analysis": {
                "sensitive_words": analysis_result.get("sensitive_words", []),
            },
            "tags": analysis_result.get("tags", {}),
        }

    except Exception as e:

        print("\n====== PROCESS ERROR ======\n")
        print(str(e))
        print("\n===========================\n")

        return {
            "success": False,
            "error": str(e)
        }