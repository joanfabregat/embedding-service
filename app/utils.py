#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

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
