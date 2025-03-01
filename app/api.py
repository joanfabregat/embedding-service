# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import VERSION, BUILD_ID, EMBEDDING_MODEL, COMMIT_SHA
from app.embedders import load_embedder
from app.logging import logger

if not EMBEDDING_MODEL:
    raise ValueError("No embedding model specified")

logger.info(f"Starting Embedding Service {VERSION} ({BUILD_ID})")
startup = datetime.now()
embedder = load_embedder(EMBEDDING_MODEL)
api = FastAPI(
    title="Embedding Service",
    version=VERSION,
    description=f"API for generating embeddings using the model {embedder.MODEL_NAME}",
)


class RootResponse(BaseModel):
    """Response schema for root endpoint"""
    version: str
    build_id: str
    commit_sha: str
    uptime: float
    embedding_model: str
    device: str


@api.get("/", response_model=RootResponse, tags=["root"])
def root():
    return RootResponse(
        version=VERSION,
        build_id=BUILD_ID,
        commit_sha=COMMIT_SHA,
        uptime=round((datetime.now() - startup).total_seconds()),
        embedding_model=embedder.MODEL_NAME,
        device=embedder.DEVICE,
    )


# Process batch of texts to embeddings
@api.post("/batch_embed", response_model=embedder.BatchEmbedResponse)
def batch_embed(request: embedder.BatchEmbedRequest) -> embedder.BatchEmbedResponse:
    """
    Get embeddings for a batch of texts
    """
    return embedder.batch_embed(request)


@api.post("/count_tokens")
def count_tokens_(request: embedder.TokensCountRequest) -> embedder.TokensCountResponse:
    """
    Count the number of tokens in a batch of texts
    """
    return embedder.count_tokens(request)
