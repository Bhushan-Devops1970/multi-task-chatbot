"""Streamlit user interface for the multi-task chatbot."""

from __future__ import annotations

import logging

import streamlit as st

from config import APP_CONFIG, MODEL_CONFIG, configure_logging
from models.llm import describe_model
from utils.helpers import (
    add_history_item,
    build_download_filename,
    export_history_text,
    initialize_session_state,
    save_text_to_file,
)
from utils.router import TaskName, route_task

logger = logging.getLogger(__name__)


COMMON_LANGUAGES = [
    "Spanish",
    "French",
    "German",
    "Hindi",
    "Chinese",
    "Japanese",
    "Arabic",
    "Portuguese",
    "Italian",
    "English",
]


def main() -> None:
    """Run the Streamlit application."""

    configure_logging()
    st.set_page_config(
        page_title=APP_CONFIG.app_name,
        page_icon="AI",
        layout="wide",
    )
    initialize_session_state()

    st.title(APP_CONFIG.app_name)
    st.caption("A modular LangChain and Hugging Face assistant for common NLP tasks.")

    temperature, max_new_tokens, save_path = render_sidebar()

    tabs = st.tabs([task.value for task in TaskName])

    with tabs[0]:
        render_generation_tab(temperature, max_new_tokens, save_path)

    with tabs[1]:
        render_summarization_tab(temperature, max_new_tokens, save_path)

    with tabs[2]:
        render_translation_tab(temperature, max_new_tokens, save_path)

    with tabs[3]:
        render_sentiment_tab(temperature, max_new_tokens, save_path)

    with tabs[4]:
        render_qa_tab(temperature, max_new_tokens, save_path)

    render_history()


def render_sidebar() -> tuple[float, int, str]:
    """Render model and export controls in the sidebar."""

    with st.sidebar:
        st.header("Model settings")
        st.info(describe_model())

        temperature = st.slider(
            "Temperature",
            min_value=MODEL_CONFIG.min_temperature,
            max_value=MODEL_CONFIG.max_temperature,
            value=MODEL_CONFIG.default_temperature,
            step=0.05,
            help="Higher values produce more varied responses.",
        )
        max_new_tokens = st.slider(
            "Max tokens",
            min_value=MODEL_CONFIG.min_new_tokens,
            max_value=MODEL_CONFIG.max_new_tokens,
            value=MODEL_CONFIG.default_max_new_tokens,
            step=32,
            help="Maximum number of generated tokens.",
        )

        st.divider()
        st.header("Session")
        save_path = st.text_input("Save file", value=APP_CONFIG.history_file)

        if st.button("Save full history", use_container_width=True):
            try:
                path = save_text_to_file(export_history_text(), save_path)
                st.success(f"Saved to {path}")
            except Exception as exc:
                logger.exception("Saving full history failed")
                st.error(str(exc))

        st.download_button(
            label="Export history as txt",
            data=export_history_text(),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.button("Clear session memory", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_response = ""
            st.rerun()

    return temperature, max_new_tokens, save_path


def render_generation_tab(
    temperature: float,
    max_new_tokens: int,
    save_path: str,
) -> None:
    """Render the text generation workflow."""

    st.subheader(TaskName.GENERATION.value)
    prompt = st.text_area(
        "Prompt",
        height=180,
        placeholder="Ask for an explanation, draft, plan, or detailed answer.",
    )

    if st.button("Generate response", key="generate_button"):
        run_task(
            task_name=TaskName.GENERATION,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_path=save_path,
            inputs={"prompt": prompt},
        )


def render_summarization_tab(
    temperature: float,
    max_new_tokens: int,
    save_path: str,
) -> None:
    """Render the summarization workflow."""

    st.subheader(TaskName.SUMMARIZATION.value)
    text = st.text_area(
        "Text to summarize",
        height=260,
        placeholder="Paste long paragraphs, notes, articles, or transcripts.",
    )

    if st.button("Summarize", key="summarize_button"):
        run_task(
            task_name=TaskName.SUMMARIZATION,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_path=save_path,
            inputs={"text": text},
        )


def render_translation_tab(
    temperature: float,
    max_new_tokens: int,
    save_path: str,
) -> None:
    """Render the translation workflow."""

    st.subheader(TaskName.TRANSLATION.value)
    col_language, col_custom = st.columns([1, 2])
    with col_language:
        selected_language = st.selectbox("Target language", COMMON_LANGUAGES)
    with col_custom:
        custom_language = st.text_input(
            "Custom target language",
            placeholder="Optional, e.g. Korean or Tamil",
        )

    target_language = custom_language.strip() or selected_language
    text = st.text_area(
        "Text to translate",
        height=220,
        placeholder="Enter text to translate.",
    )

    if st.button("Translate", key="translate_button"):
        run_task(
            task_name=TaskName.TRANSLATION,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_path=save_path,
            inputs={"text": text, "target_language": target_language},
        )


def render_sentiment_tab(
    temperature: float,
    max_new_tokens: int,
    save_path: str,
) -> None:
    """Render the sentiment analysis workflow."""

    st.subheader(TaskName.SENTIMENT.value)
    text = st.text_area(
        "Text to analyze",
        height=180,
        placeholder="Enter customer feedback, reviews, messages, or social posts.",
    )

    if st.button("Analyze sentiment", key="sentiment_button"):
        run_task(
            task_name=TaskName.SENTIMENT,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_path=save_path,
            inputs={"text": text},
        )


def render_qa_tab(
    temperature: float,
    max_new_tokens: int,
    save_path: str,
) -> None:
    """Render the context-based QA workflow."""

    st.subheader(TaskName.QA.value)
    context = st.text_area(
        "Context",
        height=220,
        placeholder="Paste the source context the model should use.",
    )
    question = st.text_input(
        "Question",
        placeholder="Ask a question that can be answered from the context.",
    )

    if st.button("Answer question", key="qa_button"):
        run_task(
            task_name=TaskName.QA,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_path=save_path,
            inputs={"context": context, "question": question},
        )


def run_task(
    task_name: TaskName,
    temperature: float,
    max_new_tokens: int,
    save_path: str,
    inputs: dict[str, str],
) -> None:
    """Execute a task and render its response controls."""

    try:
        with st.spinner("Running model..."):
            response = route_task(
                task_name=task_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **inputs,
            )

        add_history_item(task=task_name.value, inputs=inputs, response=response)
        render_response(task_name=task_name, response=response, save_path=save_path)
    except Exception as exc:
        st.error(str(exc))


def render_response(task_name: TaskName, response: str, save_path: str) -> None:
    """Render a model response and export actions."""

    st.markdown("### Response")
    st.write(response)

    col_save, col_download = st.columns([1, 1])
    with col_save:
        if st.button("Save latest response", key=f"save_{task_name.value}"):
            try:
                path = save_text_to_file(response, save_path)
                st.success(f"Saved to {path}")
            except Exception as exc:
                logger.exception("Saving latest response failed")
                st.error(str(exc))

    with col_download:
        st.download_button(
            label="Export result as txt",
            data=response,
            file_name=build_download_filename(task_name.value),
            mime="text/plain",
            key=f"download_{task_name.value}",
        )


def render_history() -> None:
    """Render chat history from session memory."""

    st.divider()
    st.subheader("Chat history")

    history = st.session_state.get("history", [])
    if not history:
        st.caption("No interactions in this session yet.")
        return

    for index, item in enumerate(reversed(history), start=1):
        title = f"{item['task']} - {item['timestamp']}"
        with st.expander(title, expanded=index == 1):
            st.markdown("**Inputs**")
            for key, value in item["inputs"].items():
                st.text_area(
                    key.replace("_", " ").title(),
                    value=value,
                    height=100,
                    disabled=True,
                    key=f"history_{index}_{key}",
                )
            st.markdown("**Response**")
            st.write(item["response"])
