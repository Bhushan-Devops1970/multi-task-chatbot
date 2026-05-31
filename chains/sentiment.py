"""Sentiment analysis chain."""

from __future__ import annotations

import logging

from langchain.chains import LLMChain

from models.llm import get_llm
from prompts.sentiment_prompt import SENTIMENT_PROMPT

logger = logging.getLogger(__name__)


VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def analyze_sentiment(text: str, temperature: float, max_new_tokens: int) -> str:
    """Classify text as Positive, Negative, or Neutral."""

    if not text.strip():
        raise ValueError("Please enter text for sentiment analysis.")

    try:
        chain = LLMChain(
            llm=get_llm(temperature=temperature, max_new_tokens=max_new_tokens),
            prompt=SENTIMENT_PROMPT,
        )
        response = chain.run(source_text=text.strip()).strip()
        normalized = response.lower()

        for sentiment in VALID_SENTIMENTS:
            if sentiment in normalized:
                return sentiment.title()

        return f"Neutral\n\nModel note: {response}"
    except Exception:
        logger.exception("Sentiment analysis failed")
        raise
