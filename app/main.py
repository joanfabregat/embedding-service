#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from datetime import datetime

from fastapi import FastAPI

from app.config import Config
from app.embedders import EMBEDDERS_MAPPING, load_embedder
from app.logging import logger
from app.models import RootResponse, BatchEmbedRequest, BatchEmbedResponse, TokensCountRequest, TokensCountResponse
from app.utils import get_device

startup = datetime.now()

logger.info(f"Starting Embedding Service {Config.APP_VERSION} ({Config.APP_BUILD_ID})")
app = FastAPI(
    title="Embedding Service",
    version=Config.APP_VERSION,
    description="API for generating sparse and dense embeddings from text"
)


@app.get("/", response_model=RootResponse, tags=["root"])
def root():
    return RootResponse(
        version=Config.APP_VERSION,
        build_id=Config.APP_BUILD_ID,
        commit_sha=Config.APP_COMMIT_SHA,
        uptime=(datetime.now() - startup).total_seconds(),
        embedders=[embedder for embedder in EMBEDDERS_MAPPING.keys()],
        device=get_device(),
    )


# Process batch of texts to embeddings
@app.post("/batch_embed", response_model=BatchEmbedResponse)
def batch_embed(request: BatchEmbedRequest) -> BatchEmbedResponse:
    """
    Get embeddings for a batch of texts

    Args:
        request: The request containing the texts

    Returns:
        BatchEmbedResponse: The embeddings of the texts
    """
    start = datetime.now()

    # Convert to numpy and then to list for JSON serialization
    embedder = load_embedder(request.embedder)
    embeddings = embedder.batch_embed(request.texts, normalize=request.normalize, task=request.task)
    return BatchEmbedResponse(
        embeddings=embeddings,
        count=len(embeddings),
        dimensions=len(embeddings[0]) if embeddings else 0,
        compute_time=(datetime.now() - start).total_seconds()

    )


@app.post("/count_tokens")
def count_tokens_(request: TokensCountRequest) -> TokensCountResponse:
    """
    Count the number of tokens in a batch of texts

    Args:
        request: The request containing

    Returns:
        TokensCountResponse: The number of tokens in each text
    """
    start = datetime.now()
    embedder = load_embedder(request.embedder)
    return TokensCountResponse(
        tokens_count=[embedder.count_tokens(text) for text in request.texts],
        compute_time=(datetime.now() - start).total_seconds()
    )
