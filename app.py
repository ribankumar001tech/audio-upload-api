# # app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import tempfile
import requests
import os

from services.audio_processor import process_audio


app = Flask(__name__)
CORS(app)

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

    temp_file_path = None

    try:

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

            # create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{extension}"
            )

            file.save(temp_file.name)

            temp_file_path = temp_file.name

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

            response = requests.get(audio_url, timeout=60)

            if response.status_code != 200:
                return jsonify({
                    "success": False,
                    "message": "Failed to download audio"
                }), 400

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )

            temp_file.write(response.content)
            temp_file.close()

            temp_file_path = temp_file.name

        else:

            return jsonify({
                "success": False,
                "message": "Upload file OR send audio_url"
            }), 400

        # =====================================================
        # PROCESS AUDIO
        # =====================================================

        result = process_audio(temp_file_path)

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

        final_response = {
            "job_id": job_id,
            "status": "completed",
            "transcript": result.get("transcript", []),
            "tags": result.get("tags", {})
        }

        return jsonify(final_response)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        # =====================================================
        # DELETE TEMP FILE
        # =====================================================

        try:

            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as cleanup_error:

            print("FILE DELETE ERROR:", cleanup_error)


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )