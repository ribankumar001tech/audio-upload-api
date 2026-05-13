# services/audio_processor.py

from services.sarvam_service import transcribe_audio
from services.context_ollama import analyze_transcription

def process_audio(audio_path):
    try:
        transcription_result = transcribe_audio(audio_path)
        transcript_text = transcription_result.get("transcript", "")

        if not transcript_text:
            raise Exception("Empty transcript returned from Sarvam")

        analysis_result = analyze_transcription(transcript_text)

        return {
            "transcript": transcript_text,
            "analysis": analysis_result
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }