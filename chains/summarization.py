"""Text summarization chain."""

from __future__ import annotations

import logging

from models.llm import generate_from_prompt
from prompts.summary_prompt import SUMMARY_PROMPT
from utils.helpers import chunk_text

logger = logging.getLogger(__name__)


def summarize_text(text: str, temperature: float, max_new_tokens: int) -> str:
    """Summarize long text, chunking large paragraphs when needed."""

    if not text.strip():
        raise ValueError("Please enter text to summarize.")

    try:
        chunks = chunk_text(text.strip(), max_words=350)
        summaries = [
            generate_from_prompt(
                SUMMARY_PROMPT.format(source_text=chunk),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            for chunk in chunks
            if chunk.strip()
        ]

        if len(summaries) == 1:
            return summaries[0]

        combined = "\n".join(summaries)
        return generate_from_prompt(
            SUMMARY_PROMPT.format(source_text=combined),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception("Summarization failed")
        raise
