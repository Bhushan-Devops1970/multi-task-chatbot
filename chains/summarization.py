"""Text summarization chain."""

from __future__ import annotations

import logging

from langchain.chains import LLMChain

from models.llm import get_llm
from prompts.summary_prompt import SUMMARY_PROMPT
from utils.helpers import chunk_text

logger = logging.getLogger(__name__)


def summarize_text(text: str, temperature: float, max_new_tokens: int) -> str:
    """Summarize long text, chunking large paragraphs when needed."""

    if not text.strip():
        raise ValueError("Please enter text to summarize.")

    try:
        chain = LLMChain(
            llm=get_llm(temperature=temperature, max_new_tokens=max_new_tokens),
            prompt=SUMMARY_PROMPT,
        )
        chunks = chunk_text(text.strip(), max_words=350)
        summaries = [
            chain.run(source_text=chunk).strip()
            for chunk in chunks
            if chunk.strip()
        ]

        if len(summaries) == 1:
            return summaries[0]

        combined = "\n".join(summaries)
        return chain.run(source_text=combined).strip()
    except Exception:
        logger.exception("Summarization failed")
        raise
