#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from app.logger import logger
from .base_embedder import BaseEmbedder


class BaseTransformerEmbedder(BaseEmbedder):
    """
    A dense embedder that uses a transformer model to create embeddings.
    """
    model_name: str = ...
    tokenizer: PreTrainedTokenizer | None = ...
    model: PreTrainedModel | None = ...

    @staticmethod
    def get_device(allow_gpu: bool) -> torch.device:
        if allow_gpu and torch.cuda.is_available():
            logger.info("Using GPU for as Torch device")
            return torch.device("cuda")
        if allow_gpu and torch.backends.mps.is_available():
            logger.info("Using MPS for as Torch device")
            return torch.device("mps")
        else:
            logger.info("Using CPU for as Torch device")
            return torch.device("cpu")
