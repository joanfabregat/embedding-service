#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import enum

from pydantic import BaseModel


class RootResponse(BaseModel):
    """Response schema for root endpoint"""
    status: str = "ok"
    version: str
    build_id: str
    commit_sha: str
    uptime: float
    device: str


class ModelType(str, enum.Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class ModelCard(BaseModel):
    model_name: str
    description: str
    url: str
    max_tokens: int
    type: ModelType


class BatchEmbedRequest(BaseModel):
    """Request schema for embeddings"""
    texts: list[str]


class BatchEmbedResponse(BaseModel):
    """Response schema for embeddings"""
    embeddings: list[list[float]]
    count: int
    dimensions: int
    compute_time: float


class TokensCountRequest(BaseModel):
    """Request schema for tokens count"""
    texts: list[str]


class TokensCountResponse(BaseModel):
    """Response schema for tokens count"""
    tokens_count: list[int]
    compute_time: float
