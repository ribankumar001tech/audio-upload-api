# services/ollama_service.py

import json
import urllib.request
import urllib.error
from config.settings import OLLAMA_CONFIG


def call_ollama(prompt: str):
    """
    Send prompt to Ollama and return JSON response
    """

    try:
        payload = {
            "model": OLLAMA_CONFIG["model_name"],
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 4000,
                "num_ctx": 8192
            }
        }

        req = urllib.request.Request(
            OLLAMA_CONFIG["api_url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(
            req,
            timeout=OLLAMA_CONFIG["timeout_seconds"]
        ) as response:

            result = response.read().decode("utf-8")

        parsed = json.loads(result)

        if "response" not in parsed:
            return {
                "success": False,
                "error": "No response from Ollama"
            }

        model_output = parsed["response"]

        # Try parsing model JSON output
        try:
            final_json = json.loads(model_output)

            return {
                "success": True,
                "data": final_json
            }

        except Exception:
            return {
                "success": False,
                "error": "Model returned invalid JSON",
                "raw_output": model_output
            }

    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": f"Ollama connection error: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }