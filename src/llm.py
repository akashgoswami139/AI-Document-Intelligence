from __future__ import annotations

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_groq_model() -> ChatGroq:
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )