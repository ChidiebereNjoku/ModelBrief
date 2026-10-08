import os

from .base import AIProvider


class GroqProvider(AIProvider):

    def __init__(self, api_key=None, model=None):
        key = api_key or os.getenv("GROQ_API_KEY")

        if not key:
            raise ValueError("GROQ_API_KEY is required when ai=True")

        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                'Install AI support with: pip install "modelbrief[ai]"'
            ) from e

        self.client = Groq(api_key=key)

        self.model = model or os.getenv(
            "MODELBRIEF_GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    def complete(self, system, user):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )

        return response.choices[0].message.content