#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from pydantic import BaseModel

from app.embedders import AvailableEmbedders


class RootResponse(BaseModel):
    """Response schema for root endpoint"""
    status: str = "ok"
    version: str
    build_id: str
    commit_sha: str
    uptime: float
    embedders: list[str]
    device: str


class BatchEmbedRequest(BaseModel):
    """Request schema for embeddings"""
    embedder: AvailableEmbedders
    texts: list[str]
    normalize: bool = True
    task: str = None


class BatchEmbedResponse(BaseModel):
    """Response schema for embeddings"""
    embeddings: list[list[float] | tuple[list[int], list[float]]]
    count: int
    dimensions: int
    compute_time: float


class TokensCountRequest(BaseModel):
    """Request schema for tokens count"""
    embedder: AvailableEmbedders
    texts: list[str]


class TokensCountResponse(BaseModel):
    """Response schema for tokens count"""
    tokens_count: list[int]
    compute_time: float
