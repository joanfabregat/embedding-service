#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

from typing import TypeVar, Generic

from pydantic import BaseModel

DenseVector = list[float]
SparseVector = tuple[list[int], list[float]]

class RootResponse(BaseModel):
    """Response schema for root endpoint"""
    version: str
    build_id: str
    commit_sha: str
    uptime: float
    embedding_model: str
    device: str

SettingsType = TypeVar('SettingsType')

class BatchEmbedRequest(BaseModel, Generic[SettingsType]):
    """Request schema for embeddings"""
    texts: list[str]
    settings: SettingsType


class BatchEmbedResponse(BaseModel):
    """Response schema for embeddings"""
    model_name: str
    embeddings: list[DenseVector | SparseVector]
    count: int
    dimensions: int
    compute_time: float


class TokensCountRequest(BaseModel):
    """Request schema for tokens count"""
    texts: list[str]


class TokensCountResponse(BaseModel):
    """Response schema for tokens count"""
    model_name: str
    tokens_count: list[int]
    compute_time: float
