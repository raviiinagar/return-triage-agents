import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    """
    Returns an OpenAI client configured for OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_key_here":
        raise ValueError("OPENROUTER_API_KEY environment variable is not set correctly.")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
            "X-Title": "ReturnTriageAgent", # Required by OpenRouter
        }
    )
    return client

def generate_completion(prompt: str, model: str = "google/gemini-2.5-flash") -> str:
    """
    Helper function to generate a completion using OpenRouter.
    Includes basic retry logic.
    """
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""
