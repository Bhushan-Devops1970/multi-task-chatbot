"""Context-based question answering chain."""

from __future__ import annotations

import logging

from langchain.chains import LLMChain

from models.llm import get_llm
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
        chain = LLMChain(
            llm=get_llm(temperature=temperature, max_new_tokens=max_new_tokens),
            prompt=QA_PROMPT,
        )
        return chain.run(
            context=context.strip(),
            question=question.strip(),
        ).strip()
    except Exception:
        logger.exception("Question answering failed")
        raise
