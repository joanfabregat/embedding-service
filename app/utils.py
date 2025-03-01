#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

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
