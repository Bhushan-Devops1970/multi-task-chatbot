"""Model loading utilities for Hugging Face Transformers."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL_CONFIG, ModelConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model_components() -> tuple[Any, Any, torch.device]:
    """Create and cache the local Hugging Face model, tokenizer, and device."""

    return get_seq2seq_components(MODEL_CONFIG.model_name)


@lru_cache(maxsize=4)
def get_seq2seq_components(model_name: str) -> tuple[Any, Any, torch.device]:
    """Create and cache a Hugging Face seq2seq model, tokenizer, and device."""

    logger.info(
        "Loading model=%s device=%s",
        model_name,
        MODEL_CONFIG.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device(
        f"cuda:{MODEL_CONFIG.device}"
        if MODEL_CONFIG.device >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_from_prompt(
    prompt: str,
    temperature: float = MODEL_CONFIG.default_temperature,
    max_new_tokens: int = MODEL_CONFIG.default_max_new_tokens,
) -> str:
    """Run the shared text2text model with bounded generation settings."""

    return generate_with_seq2seq_model(
        model_name=MODEL_CONFIG.model_name,
        input_text=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def generate_with_seq2seq_model(
    model_name: str,
    input_text: str,
    temperature: float = MODEL_CONFIG.default_temperature,
    max_new_tokens: int = MODEL_CONFIG.default_max_new_tokens,
) -> str:
    """Generate text with a specific cached seq2seq model."""

    safe_temperature = max(
        MODEL_CONFIG.min_temperature,
        min(temperature, MODEL_CONFIG.max_temperature),
    )
    safe_max_new_tokens = max(
        MODEL_CONFIG.min_new_tokens,
        min(max_new_tokens, MODEL_CONFIG.max_new_tokens),
    )

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": safe_max_new_tokens,
    }
    if safe_temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = max(safe_temperature, 0.01)
    else:
        generation_kwargs["do_sample"] = False

    tokenizer, model, device = get_seq2seq_components(model_name)
    encoded_prompt = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**encoded_prompt, **generation_kwargs)

    response = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    if not response:
        raise RuntimeError("The model returned an empty response.")

    return response


def describe_model(config: ModelConfig = MODEL_CONFIG) -> str:
    """Return a concise model description for the UI."""

    device_label = "CPU" if config.device < 0 else f"CUDA device {config.device}"
    return f"{config.model_name} on {device_label}"
