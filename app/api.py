#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from datetime import datetime

from fastapi import FastAPI

from app.config import Config
from app.embedders import load_embedder
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
        uptime=round((datetime.now() - startup).total_seconds()),
        available_models=Config.ENABLED_MODELS,
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
    embedder = load_embedder(model_name=request.model)
    embeddings = embedder.batch_embed(request.texts, config=request.config)
    return BatchEmbedResponse(
        model=request.model,
        embeddings=embeddings,
        count=len(embeddings),
        dimensions=(
            len(embeddings[0][0])
            if isinstance(embeddings[0], tuple)
            else len(embeddings[0])
            if embeddings else 0
        ),
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
    embedder = load_embedder(model_name=request.model)
    return TokensCountResponse(
        model=request.model,
        tokens_count=[embedder.count_tokens(text) for text in request.texts],
        compute_time=(datetime.now() - start).total_seconds()
    )
