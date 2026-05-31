"""Helper utilities for session memory, chunking, and exports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


HistoryItem = dict[str, Any]


def initialize_session_state() -> None:
    """Initialize Streamlit session state used by the application."""

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_response", "")


def add_history_item(
    task: str,
    inputs: dict[str, str],
    response: str,
) -> None:
    """Store an interaction in session memory."""

    item: HistoryItem = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "inputs": inputs,
        "response": response,
    }
    st.session_state.history.append(item)
    st.session_state.last_response = response


def format_history_item(item: HistoryItem) -> str:
    """Format a history entry for display or export."""

    inputs = "\n".join(
        f"{key}: {value}" for key, value in item.get("inputs", {}).items()
    )
    return (
        f"Time: {item.get('timestamp', '')}\n"
        f"Task: {item.get('task', '')}\n"
        f"{inputs}\n"
        f"Response:\n{item.get('response', '')}\n"
    )


def export_history_text() -> str:
    """Return the current session history as plain text."""

    history = st.session_state.get("history", [])
    if not history:
        return "No chat history available."

    separator = "\n" + "-" * 80 + "\n"
    return separator.join(format_history_item(item) for item in history)


def save_text_to_file(content: str, file_path: str) -> Path:
    """Save text content to a UTF-8 file."""

    if not content.strip():
        raise ValueError("No content available to save.")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Saved response content to %s", path)
    return path


def chunk_text(text: str, max_words: int = 350) -> list[str]:
    """Split long text into word chunks for model-friendly summarization."""

    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    for index in range(0, len(words), max_words):
        chunks.append(" ".join(words[index : index + max_words]))
    return chunks


def build_download_filename(task_name: str) -> str:
    """Create a stable text export filename for a task response."""

    safe_task_name = task_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_task_name}_{timestamp}.txt"
