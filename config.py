"""Application configuration and defaults."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Runtime configuration for the Hugging Face text2text model."""

    model_name: str = os.getenv("MODEL_NAME", "google/flan-t5-small")
    device: int = int(os.getenv("MODEL_DEVICE", "-1"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    default_max_new_tokens: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "256"))
    min_temperature: float = 0.0
    max_temperature: float = 1.5
    min_new_tokens: int = 32
    max_new_tokens: int = 512


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration."""

    app_name: str = "Multi Task Chatbot"
    history_file: str = os.getenv("CHAT_HISTORY_FILE", "chat_history.txt")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


MODEL_CONFIG = ModelConfig()
APP_CONFIG = AppConfig()


def configure_logging() -> None:
    """Configure process-wide logging once."""

    logging.basicConfig(
        level=getattr(logging, APP_CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
