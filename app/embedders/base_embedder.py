# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from pydantic import BaseModel

SparseVector = tuple[list[int], list[float]]
DenseVector = list[float]


class BaseEmbedder:
    """Base class for embedders"""
    MODEL_NAME = ...

    class BatchEmbedRequest(BaseModel):
        """Request schema for embeddings"""
        texts: list[str]

    class BatchEmbedResponse(BaseModel):
        """Response schema for embeddings"""
        model: str
        embeddings: list[DenseVector | SparseVector]

        @property
        def count(self) -> int:
            return len(self.embeddings)

        @property
        def dimensions(self) -> int:
            return (
                len(self.embeddings[0][0])
                if isinstance(self.embeddings[0], tuple)
                else len(self.embeddings[0])
                if self.embeddings else 0
            )

    class TokensCountRequest(BaseModel):
        """Request schema for tokens count"""
        texts: list[str]

    class TokensCountResponse(BaseModel):
        """Response schema for tokens count"""
        model: str
        tokens_count: list[int]

    def batch_embed(self, request: BatchEmbedRequest) -> BatchEmbedResponse:
        """Embed a batch of texts."""
        raise NotImplementedError

    def count_tokens(self, request: TokensCountRequest) -> TokensCountResponse:
        """Count the number of tokens in a text."""
        raise NotImplementedError
