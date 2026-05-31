"""Translation chain."""

from __future__ import annotations

import logging

from models.llm import generate_from_prompt, generate_with_seq2seq_model
from prompts.translation_prompt import TRANSLATION_PROMPT

logger = logging.getLogger(__name__)


TRANSLATION_MODELS = {
    "arabic": "Helsinki-NLP/opus-mt-en-ar",
    "chinese": "Helsinki-NLP/opus-mt-en-zh",
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "hindi": "Helsinki-NLP/opus-mt-en-hi",
    "italian": "Helsinki-NLP/opus-mt-en-it",
    "japanese": "Helsinki-NLP/opus-mt-en-jap",
    "portuguese": "Helsinki-NLP/opus-mt-en-ROMANCE",
    "spanish": "Helsinki-NLP/opus-mt-en-es",
}


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
        normalized_language = target_language.strip().lower()
        translation_model = TRANSLATION_MODELS.get(normalized_language)
        if translation_model:
            return generate_with_seq2seq_model(
                model_name=translation_model,
                input_text=text.strip(),
                temperature=0.0,
                max_new_tokens=max_new_tokens,
            )

        model_prompt = TRANSLATION_PROMPT.format(
            source_text=text.strip(),
            target_language=target_language.strip(),
        )
        return generate_from_prompt(
            model_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception("Translation failed")
        raise
