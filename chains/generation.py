"""Text generation chain."""

from __future__ import annotations

import logging

from langchain.chains import LLMChain

from models.llm import get_llm
from prompts.generation_prompt import GENERATION_PROMPT

logger = logging.getLogger(__name__)


def generate_text(prompt: str, temperature: float, max_new_tokens: int) -> str:
    """Generate a detailed response for a user prompt."""

    if not prompt.strip():
        raise ValueError("Please enter a prompt for text generation.")

    try:
        chain = LLMChain(
            llm=get_llm(temperature=temperature, max_new_tokens=max_new_tokens),
            prompt=GENERATION_PROMPT,
        )
        return chain.run(user_prompt=prompt.strip()).strip()
    except Exception:
        logger.exception("Text generation failed")
        raise
