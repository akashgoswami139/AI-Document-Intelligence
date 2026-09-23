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

        data = response.json()["data"]

        return [item["embedding"] for item in data]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]