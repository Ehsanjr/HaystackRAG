
import requests
from haystack import component


@component
class OllamaGenerator:
    """Generate an answer using a local Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2:latest",
        url: str = "http://localhost:11434/api/chat",
    ):
        self.model = model
        self.url = url

    @component.output_types(reply=str)
    def run(self, prompt: str) -> dict:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer in Persian when the user asks in Persian. "
                            "When answering questions about documents, "
                            "use the provided context and do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        return {
            "reply": data["message"]["content"].strip()
        }
