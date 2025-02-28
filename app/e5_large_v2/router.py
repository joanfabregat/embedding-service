#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from datetime import datetime

from fastapi import APIRouter

from app.models import BatchEmbedResponse, TokensCountRequest, TokensCountResponse
from app.models import ModelCard, ModelType
from .embedding import batch_embed, count_tokens
from .models import E5BatchEmbedRequest

router = APIRouter(
    prefix="/e5_large_v2",
    tags=["e5_large_v2"],
)


@router.get("/model_card", response_model=ModelCard)
def model_card():
    return ModelCard(
        model_name="intfloat/multilingual-e5-large",
        type=ModelType.DENSE,
        url="https://huggingface.co/intfloat/multilingual-e5-large",
        description="A flexible embedder for the Multilingual E5 model.",
        max_tokens=512,
    )


# Process batch of texts to embeddings
@router.post("/batch_embed", response_model=BatchEmbedResponse)
def batch_embed_(request: E5BatchEmbedRequest) -> BatchEmbedResponse:
    """
    Get embeddings for a batch of texts

    Args:
        request: The request containing the texts

    Returns:
        BatchEmbedResponse: The embeddings of the texts
    """
    start = datetime.now()
    embeddings = batch_embed(request.texts, request.normalize)
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
