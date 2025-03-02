# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from datetime import datetime

from fastapi import FastAPI

from app.config import VERSION, BUILD_ID, EMBEDDING_MODEL, COMMIT_SHA
from app.embedders import load_embedder
from app.logging import logger
from app.models import RootResponse, BatchEmbedRequest, BatchEmbedResponse, TokensCountRequest, TokensCountResponse

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


BATCH_SIZE = 100


# Process batch of texts to embeddings
@api.post("/batch_embed", response_model=BatchEmbedResponse)
def batch_embed(request: BatchEmbedRequest[embedder.Settings]) -> BatchEmbedResponse:
    """
    Get embeddings for a batch of texts
    """
    logger.info(f"Embedding {len(request.texts)} texts")
    start_time = datetime.now()

    # If the batch is small enough, embed it in one go
    if len(request.texts) <= BATCH_SIZE:
        embeddings = embedder.batch_embed(request.texts, request.settings)

    # Else, split the batch into smaller batches and embed them separately
    else:
        embeddings = []
        for i in range(0, len(request.texts), BATCH_SIZE):
            batch_texts = request.texts[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i // BATCH_SIZE + 1} of {(len(request.texts) - 1) // BATCH_SIZE + 1}")
            batch_embeddings = embedder.batch_embed(batch_texts, request.settings)
            embeddings.extend(batch_embeddings)

    response = BatchEmbedResponse(
        embedding_model=embedder.MODEL_NAME,
        embeddings=embeddings,
        compute_time=(datetime.now() - start_time).total_seconds()
    )
    logger.info(f"Computed embeddings in {response.compute_time:.2f}s")
    return response


@api.post("/count_tokens")
def count_tokens_(request: TokensCountRequest) -> TokensCountResponse:
    """
    Count the number of tokens in a batch of texts
    """
    logger.info(f"Counting {len(request.texts)} tokens")
    start_time = datetime.now()
    tokens_count = embedder.batch_count_tokens(request.texts)
    response = TokensCountResponse(
        embedding_model=embedder.MODEL_NAME,
        tokens_count=tokens_count,
        compute_time=(datetime.now() - start_time).total_seconds(),
    )
    logger.info(f"Computed tokens in {response.compute_time:.2f}s")
    return response
