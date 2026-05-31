"""Sentiment analysis chain."""

from __future__ import annotations

import logging

from models.llm import generate_from_prompt
from prompts.sentiment_prompt import SENTIMENT_PROMPT

logger = logging.getLogger(__name__)


VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def analyze_sentiment(text: str, temperature: float, max_new_tokens: int) -> str:
    """Classify text as Positive, Negative, or Neutral."""

    if not text.strip():
        raise ValueError("Please enter text for sentiment analysis.")

    try:
        response = generate_from_prompt(
            SENTIMENT_PROMPT.format(source_text=text.strip()),
            temperature=0.0,
            max_new_tokens=min(max_new_tokens, 96),
        )
        normalized = response.lower()

        for sentiment in VALID_SENTIMENTS:
            if sentiment in normalized:
                return sentiment.title()

        return f"Neutral\n\nModel note: {response}"
    except Exception:
        logger.exception("Sentiment analysis failed")
        raise
