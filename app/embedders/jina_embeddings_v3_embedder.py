# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import enum

import torch
from transformers import AutoTokenizer, AutoModel

from app.logging import logger
from app.utils import get_device
from .base_embedder import BaseEmbedder, DenseVector


class JinaEmbeddingsV3Embedder(BaseEmbedder):
    """
    Embedder using the Jina embeddings v3 model
    https://huggingface.co/jinaai/jina-embeddings-v3
    """

    MODEL_NAME = "jinaai/jina-embeddings-v3"
    REVISION = "f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
    DEFAULT_NORMALIZE = True
    DEFAULT_TASK = "retrieval.query"
    EMBEDDING_TYPE = DenseVector

    class BatchEmbedRequest(BaseEmbedder.BatchEmbedRequest):
        class Task(str, enum.Enum):
            RETRIEVAL_QUERY = "retrieval.query"
            RETRIEVAL_PASSAGE = "retrieval.passage"
            SEPARATION = "separation"
            CLASSIFICATION = "classification"
            TEXT_MATCHING = "text-matching"

        normalize: bool = True
        task: Task = Task.RETRIEVAL_QUERY

    class BatchEmbedResponse(BaseEmbedder.BatchEmbedResponse):
        embeddings: list[DenseVector]

    class TokensCountRequest(BaseEmbedder.TokensCountRequest):
        pass

    class TokensCountResponse(BaseEmbedder.TokensCountResponse):
        pass

    def __init__(self):
        """Initialize the embedder."""
        logger.info(f"Initializing Jina embeddings v3 embedder with model {self.MODEL_NAME}")
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.REVISION
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.REVISION
        ).to(self.device)

    # noinspection DuplicatedCode
    def batch_embed(self, request: BatchEmbedRequest) -> BatchEmbedResponse:
        """Get embeddings for a batch of texts"""
        logger.info(f"Embedding {len(request.texts)} texts using {self.MODEL_NAME}")

        # Tokenize and prepare for model
        inputs = self.tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs, task=request.task.value)

        # Apply mean pooling and optionally normalize
        token_embeddings = model_output[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        if request.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return self.BatchEmbedResponse(
            model=self.MODEL_NAME,
            embeddings=embeddings.cpu().numpy().tolist()
        )

    def count_tokens(self, request: TokensCountRequest) -> TokensCountResponse:
        """Count the number of tokens in a text."""
        return self.TokensCountResponse(
            model=self.MODEL_NAME,
            tokens_count=[len(self.tokenizer.tokenize(text)) for text in request.texts]
        )
