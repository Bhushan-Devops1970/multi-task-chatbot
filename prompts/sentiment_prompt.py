"""Prompt template for sentiment analysis."""

from langchain.prompts import PromptTemplate


SENTIMENT_PROMPT = PromptTemplate(
    input_variables=["source_text"],
    template=(
        "Classify the sentiment of the following text as exactly one of: "
        "Positive, Negative, Neutral. Return the label first, followed by a "
        "one-sentence reason.\n\n"
        "Text:\n{source_text}\n\n"
        "Sentiment:"
    ),
)
