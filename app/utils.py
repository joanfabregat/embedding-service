# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import functools

import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel

from app.logging import logger


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """
    Get the device to run the model on

    Returns:
        str: The device to run the model on
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.mps.is_available():
        return "mps"

    return "cpu"


def load_tokenizer_and_model(
        model_name: str,
        **kwargs
) -> tuple[PreTrainedTokenizer, PreTrainedModel, str]:
    """
    Load a model and tokenizer from the Hugging Face model hub.

    Args:
        model_name: The name of the model to load

    Returns:
        A tuple containing the tokenizer, model, and device
    """
    logger.info(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    device = get_device()
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    return tokenizer, model, device
