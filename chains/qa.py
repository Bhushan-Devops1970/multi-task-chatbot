"""Context-based question answering chain."""

from __future__ import annotations

import logging

from models.llm import generate_from_prompt
from prompts.qa_prompt import QA_PROMPT

logger = logging.getLogger(__name__)


def answer_question(
    context: str,
    question: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Answer a question using only the provided context."""

    if not context.strip():
        raise ValueError("Please enter context for question answering.")
    if not question.strip():
        raise ValueError("Please enter a question.")

    try:
        model_prompt = QA_PROMPT.format(
            context=context.strip(),
            question=question.strip(),
        )
        return generate_from_prompt(
            model_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception("Question answering failed")
        raise
