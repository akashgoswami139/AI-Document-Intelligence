
from src.jina_embeddings import JinaEmbeddings


def get_embedding_model() -> JinaEmbeddings:
    """Return the project's only embedding implementation: Jina."""
    return JinaEmbeddings()
