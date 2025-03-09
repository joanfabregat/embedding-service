# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from typing import Iterator
import torch
from pydantic import BaseModel

from app.models import DenseVector, SparseVector


class BaseEmbedder:
    """Base class for embedders"""
    MODEL_NAME: str = ...
    DEVICE: torch.device | None = None

    class Settings(BaseModel):
        pass

    def batch_embed(self, texts: list[str], settings: Settings = None) -> list[DenseVector | SparseVector | None]:
        """Embed a batch of texts."""
        raise NotImplementedError

    def batch_count_tokens(self, texts: list[str]) -> list[int]:
        """Count the number of tokens in a batch of texts."""
        return [
            self.count_tokens(text)
            for text in texts
        ]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        raise NotImplementedError

    @staticmethod
    def _create_batches(texts: list[str], batch_size: int) -> Iterator[list[str]]:
        """Split list of texts into batches of specified size."""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]
