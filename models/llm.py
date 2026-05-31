"""Model loading utilities for LangChain and Hugging Face Transformers."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config import MODEL_CONFIG, ModelConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_llm(
    temperature: float = MODEL_CONFIG.default_temperature,
    max_new_tokens: int = MODEL_CONFIG.default_max_new_tokens,
) -> HuggingFacePipeline:
    """Create and cache a LangChain-compatible Hugging Face pipeline."""

    safe_temperature = max(
        MODEL_CONFIG.min_temperature,
        min(temperature, MODEL_CONFIG.max_temperature),
    )
    safe_max_new_tokens = max(
        MODEL_CONFIG.min_new_tokens,
        min(max_new_tokens, MODEL_CONFIG.max_new_tokens),
    )

    logger.info(
        "Loading model=%s device=%s temperature=%s max_new_tokens=%s",
        MODEL_CONFIG.model_name,
        MODEL_CONFIG.device,
        safe_temperature,
        safe_max_new_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CONFIG.model_name)

    text2text_pipeline = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=MODEL_CONFIG.device,
        max_new_tokens=safe_max_new_tokens,
        do_sample=safe_temperature > 0,
        temperature=max(safe_temperature, 0.01),
    )

    return HuggingFacePipeline(pipeline=text2text_pipeline)


def describe_model(config: ModelConfig = MODEL_CONFIG) -> str:
    """Return a concise model description for the UI."""

    device_label = "CPU" if config.device < 0 else f"CUDA device {config.device}"
    return f"{config.model_name} on {device_label}"
