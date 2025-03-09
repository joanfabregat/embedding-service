# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import gc

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from app.logging import logger
from .base_embedder import BaseEmbedder
from .utils import get_computation_device


class BaseTransformerEmbedder(BaseEmbedder):
    """Base class for dense embedders."""
    DEVICE = get_computation_device()

    class Settings(BaseEmbedder.Settings):
        normalize: bool = True

    def __init__(self):
        """Initialize the embedder."""
        self.tokenizer: PreTrainedTokenizer = ...
        self.model: PreTrainedModel = ...

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.tokenize(text))

    def _move_model_to_device(self):
        """Move model to the appropriate device."""
        try:
            self.model = self.model.to('cpu')
            self.model = self.model.to(self.DEVICE)
            for param in self.model.parameters():
                if param.device.type != self.DEVICE.type:
                    param.data = param.data.to(self.DEVICE)

            logger.info(f"Successfully moved model to {self.DEVICE}")
        except Exception as e:
            logger.warning(f"Failed to move model to {self.DEVICE}, falling back to CPU: {e}")
            self.model = self.model.to('cpu')
            self.DEVICE = torch.device('cpu')

    def _move_model_to_cpu(self):
        """Move model to CPU."""
        self.model = self.model.to('cpu')
        logger.info("Moved model to CPU")

    @classmethod
    def _force_gc(cls):
        """Force garbage collection and clear CUDA/MPS cache."""
        logger.debug("Forcing garbage collection")

        gc.collect()

        if cls.DEVICE:
            match str(cls.DEVICE):
                case "cuda":
                    logger.debug("Clearing CUDA cache")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                case "mps":
                    logger.debug("Clearing MPS cache")
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
