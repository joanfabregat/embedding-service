# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from pydantic import BaseModel

SparseVector = tuple[list[int], list[float]]
DenseVector = list[float]

class RootResponse(BaseModel):
    """Response schema for root endpoint"""
    version: str
    build_id: str
    commit_sha: str
    uptime: float
    available_models: list[str]
    device: str


class BatchEmbedRequest(BaseModel):
    """Request schema for embeddings"""
    model: str
    texts: list[str]
    config: dict = None
    normalize: bool = True
    task: str = None


class BatchEmbedResponse(BaseModel):
    """Response schema for embeddings"""
    embeddings: list[list[float] | tuple[list[int], list[float]]]
    model: str
    count: int
    dimensions: int
    compute_time: float


class TokensCountRequest(BaseModel):
    """Request schema for tokens count"""
    model: str
    texts: list[str]


class TokensCountResponse(BaseModel):
    """Response schema for tokens count"""
    tokens_count: list[int]
    model: str
    compute_time: float
