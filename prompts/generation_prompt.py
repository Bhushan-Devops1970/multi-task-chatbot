"""Prompt template for text generation."""

from langchain.prompts import PromptTemplate


GENERATION_PROMPT = PromptTemplate(
    input_variables=["user_prompt"],
    template=(
        "You are a precise and helpful AI assistant. Generate a detailed, "
        "well-structured answer for the following request.\n\n"
        "Request:\n{user_prompt}\n\n"
        "Detailed answer:"
    ),
)
