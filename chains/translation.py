"""Translation chain."""

from __future__ import annotations

import logging

from langchain.chains import LLMChain

from models.llm import get_llm
from prompts.translation_prompt import TRANSLATION_PROMPT

logger = logging.getLogger(__name__)


def translate_text(
    text: str,
    target_language: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Translate text into the requested target language."""

    if not text.strip():
        raise ValueError("Please enter text to translate.")
    if not target_language.strip():
        raise ValueError("Please enter a target language.")

    try:
        chain = LLMChain(
            llm=get_llm(temperature=temperature, max_new_tokens=max_new_tokens),
            prompt=TRANSLATION_PROMPT,
        )
        return chain.run(
            source_text=text.strip(),
            target_language=target_language.strip(),
        ).strip()
    except Exception:
        logger.exception("Translation failed")
        raise
