# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base_embedder import BaseEmbedder


class BaseTransformerEmbedder(BaseEmbedder):
    """Base class for dense embedders."""
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    class Settings(BaseEmbedder.Settings):
        normalize: bool = True

    def __init__(self):
        """Initialize the embedder."""
        self.tokenizer: PreTrainedTokenizer = ...
        self.model: PreTrainedModel = ...

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.tokenize(text))
