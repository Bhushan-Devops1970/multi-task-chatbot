"""Prompt template for summarization."""

from langchain.prompts import PromptTemplate


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["source_text"],
    template=(
        "Summarize the following text clearly and concisely. Preserve the "
        "main ideas, important facts, decisions, and action items when present.\n\n"
        "Text:\n{source_text}\n\n"
        "Summary:"
    ),
)
