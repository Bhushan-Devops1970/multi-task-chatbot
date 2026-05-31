"""Prompt template for sentiment analysis."""

SENTIMENT_PROMPT = (
    "Classify the sentiment of the following text as exactly one of: "
    "Positive, Negative, Neutral. Return the label first, followed by a "
    "one-sentence reason.\n\n"
    "Text:\n{source_text}\n\n"
    "Sentiment:"
)
