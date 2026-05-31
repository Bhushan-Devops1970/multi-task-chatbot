"""Task router for chatbot operations."""

from __future__ import annotations

from enum import Enum

from chains.generation import generate_text
from chains.qa import answer_question
from chains.sentiment import analyze_sentiment
from chains.summarization import summarize_text
from chains.translation import translate_text


class TaskName(str, Enum):
    """Supported application tasks."""

    GENERATION = "Text Generation"
    SUMMARIZATION = "Summarization"
    TRANSLATION = "Translation"
    SENTIMENT = "Sentiment Analysis"
    QA = "Question Answering"


def route_task(
    task_name: TaskName,
    temperature: float,
    max_new_tokens: int,
    **kwargs: str,
) -> str:
    """Route a task request to the matching chain function."""

    if task_name == TaskName.GENERATION:
        return generate_text(
            prompt=kwargs.get("prompt", ""),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if task_name == TaskName.SUMMARIZATION:
        return summarize_text(
            text=kwargs.get("text", ""),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if task_name == TaskName.TRANSLATION:
        return translate_text(
            text=kwargs.get("text", ""),
            target_language=kwargs.get("target_language", ""),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if task_name == TaskName.SENTIMENT:
        return analyze_sentiment(
            text=kwargs.get("text", ""),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if task_name == TaskName.QA:
        return answer_question(
            context=kwargs.get("context", ""),
            question=kwargs.get("question", ""),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    raise ValueError(f"Unsupported task: {task_name}")
