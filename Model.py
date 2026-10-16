"""AI model connection for CivicLens."""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def generate_ai_response(prompt: str) -> str:
    """Generate a response using the locally running Ollama model."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException:
        return "AI service is unavailable. Please review the complaint manually."