import os
import requests
from dotenv import load_dotenv

load_dotenv()


class JinaEmbeddings:

    def __init__(self, model=None):
        self.model = model or os.getenv(
            "JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small"
        )
        self.api_key = os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in .env")

    def _embed(self, texts):
        # Changelog: convert Jina transport and malformed-response failures into actionable runtime errors.
        try:
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            embeddings = [item.get("embedding") for item in data if item.get("embedding")]
            if len(embeddings) != len(texts):
                raise ValueError("Jina returned an incomplete embedding response")
            return embeddings
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Jina embedding request failed: {exc}") from exc

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]