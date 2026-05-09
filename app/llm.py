import os
import time
import random
import requests
from google import genai

GEMINI_API_KEY = "rahasia:D"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum diset di environment variable.")

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

OLLAMA_MODEL = "qwen2.5:3b"

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

def is_retryable_gemini_error(error: Exception) -> bool:
    """
    Error yang layak dicoba ulang atau fallback:
    - 429: rate limit
    - 500: internal server error
    - 503: service unavailable / overloaded / high demand
    - 504: deadline exceeded
    """

    error_text = str(error).lower()

    retryable_keywords = [
        "429",
        "500",
        "503",
        "504",
        "resource_exhausted",
        "unavailable",
        "overloaded",
        "high demand",
        "temporarily",
        "deadline_exceeded",
        "rate limit",
        "too many requests",
    ]

    return any(keyword in error_text for keyword in retryable_keywords)


def gemini_generate_with_model(prompt: str, model: str) -> str:
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(f"Gemini model {model} tidak mengembalikan teks.")

    return response.text


def ollama_generate(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


def generate(prompt: str) -> str:
    last_error = None

    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                print(f"Trying Gemini model: {model}, attempt {attempt + 1}")
                return gemini_generate_with_model(prompt, model)

            except Exception as e:
                last_error = e

                if not is_retryable_gemini_error(e):
                    raise

                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Gemini error on {model}: {e}")
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        print(f"Switching from {model} to next fallback model...")

    print("All Gemini models failed. Falling back to Ollama...")
    print(f"Last Gemini error: {last_error}")

    return ollama_generate(prompt)


# Alias agar kode lama tetap aman
def gemini_generate(prompt: str) -> str:
    return generate(prompt)
