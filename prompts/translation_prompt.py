"""Prompt template for translation."""

from langchain.prompts import PromptTemplate


TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["source_text", "target_language"],
    template=(
        "Translate the following text into {target_language}. Keep the meaning, "
        "tone, formatting, and named entities intact. Return only the translation.\n\n"
        "Text:\n{source_text}\n\n"
        "Translation:"
    ),
)
