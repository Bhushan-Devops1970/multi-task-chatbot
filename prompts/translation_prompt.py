"""Prompt template for translation."""

TRANSLATION_PROMPT = (
    "Translate the following text into {target_language}. Keep the meaning, "
    "tone, formatting, and named entities intact. Return only the translation.\n\n"
    "Text:\n{source_text}\n\n"
    "Translation:"
)
