"""Text generation chain."""

from __future__ import annotations

import logging

from models.llm import generate_from_prompt
from prompts.generation_prompt import GENERATION_PROMPT

logger = logging.getLogger(__name__)


def generate_text(prompt: str, temperature: float, max_new_tokens: int) -> str:
    """Generate a detailed response for a user prompt."""

    if not prompt.strip():
        raise ValueError("Please enter a prompt for text generation.")

    try:
        model_prompt = GENERATION_PROMPT.format(user_prompt=prompt.strip())
        return generate_from_prompt(
            model_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception("Text generation failed")
        raise
