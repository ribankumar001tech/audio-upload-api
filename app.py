# app.py

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import os
import uuid
import requests

from services.audio_processor import process_audio

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "mp3",
    "wav",
    "m4a",
    "webm"
}


# =========================================================
# CHECK ALLOWED FILE
# =========================================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# =========================================================
# HEALTH API
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "API Running"
    })


# =========================================================
# MAIN UPLOAD API
# =========================================================

@app.route("/upload", methods=["POST"])
def upload_audio():

    try:

        file_path = None

        # =====================================================
        # GENERATE JOB ID
        # =====================================================

        job_id = str(uuid.uuid4())

        # =====================================================
        # CASE 1 → FILE UPLOAD
        # =====================================================

        if "file" in request.files:

            file = request.files["file"]

            if file.filename == "":
                return jsonify({
                    "success": False,
                    "message": "Empty filename"
                }), 400

            if not allowed_file(file.filename):
                return jsonify({
                    "success": False,
                    "message": "Invalid file type"
                }), 400

            extension = file.filename.rsplit(".", 1)[1].lower()

            filename = f"{job_id}.{extension}"

            file_path = os.path.join(
                UPLOAD_FOLDER,
                secure_filename(filename)
            )

            file.save(file_path)

        # =====================================================
        # CASE 2 → AUDIO URL
        # =====================================================

        elif request.is_json:

            data = request.get_json()

            audio_url = data.get("audio_url")

            if not audio_url:
                return jsonify({
                    "success": False,
                    "message": "audio_url is required"
                }), 400

            response = requests.get(audio_url)

            if response.status_code != 200:
                return jsonify({
                    "success": False,
                    "message": "Failed to download audio"
                }), 400

            filename = f"{job_id}.mp3"

            file_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            with open(file_path, "wb") as f:
                f.write(response.content)

        else:

            return jsonify({
                "success": False,
                "message": "Upload file OR send audio_url"
            }), 400

        # =====================================================
        # PROCESS AUDIO
        # =====================================================

        result = process_audio(file_path)

        # print("\n================ RESULT ================\n")
        # print(result)
        # print("\n========================================\n")

        # =====================================================
        # HANDLE ERRORS
        # =====================================================

        if result.get("success") is False:

            return jsonify({
                "job_id": job_id,
                "status": "failed",
                "error": result.get("error")
            }), 500

        # =====================================================
        # FINAL RESPONSE
        # =====================================================


        # Include analysis in the response (contains sensitive_words)
        final_response = {
            "job_id": job_id,
            "status": "completed",
            "transcript": result.get("transcript", []),
            "tags": result.get("tags", {}),
            # "analysis": result.get("analysis", {})
        }

        return jsonify(final_response)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )