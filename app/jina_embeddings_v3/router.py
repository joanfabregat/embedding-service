#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from datetime import datetime

from fastapi import APIRouter

from app.models import BatchEmbedResponse, TokensCountRequest, TokensCountResponse
from app.models import ModelCard, ModelType
from .embedding import get_batch_embeddings, count_tokens
from .models import JinaBatchEmbedRequest

router = APIRouter(
    prefix="/jina_embeddings_v3",
    tags=["jina_embeddings_v3"],
)


@router.get("/model_card", response_model=ModelCard)
def model_card():
    return ModelCard(
        model_name="jinaai/jina-embeddings-v3",
        type=ModelType.DENSE,
        url="https://huggingface.co/jinaai/jina-embeddings-v3",
        description="A flexible embedder for the Jina embeddings v3 model.",
        max_tokens=8192,
    )


# Process batch of texts to embeddings
@router.post("/batch_embed", response_model=BatchEmbedResponse)
def batch_embed(request: JinaBatchEmbedRequest) -> BatchEmbedResponse:
    """
    Get embeddings for a batch of texts

    Args:
        request: The request containing the texts

    Returns:
        BatchEmbedResponse: The embeddings of the texts
    """
    start = datetime.now()

    # Convert to numpy and then to list for JSON serialization
    embeddings = get_batch_embeddings(request.texts, normalize=request.normalize, task=request.task)
    return BatchEmbedResponse(
        embeddings=embeddings,
        count=len(embeddings),
        dimensions=len(embeddings[0]) if embeddings else 0,
        compute_time=(datetime.now() - start).total_seconds()

    )


@router.post("/count_tokens")
def count_tokens_(request: TokensCountRequest) -> TokensCountResponse:
    """
    Count the number of tokens in a batch of texts

    Args:
        request: The request containing

    Returns:
        TokensCountResponse: The number of tokens in each text
    """
    start = datetime.now()
    return TokensCountResponse(
        tokens_count=count_tokens(request.texts),
        compute_time=(datetime.now() - start).total_seconds()
    )
