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
from app.models import DenseVector
from .base_transformer_embedder import BaseTransformerEmbedder


class JinaEmbedder(BaseTransformerEmbedder):
    """
    Embedder using the Jina embeddings v3 model
    https://huggingface.co/jinaai/jina-embeddings-v3
    """

    MODEL_NAME = "jinaai/jina-embeddings-v3"
    MODEL_REVISION = "f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"

    class Settings(BaseTransformerEmbedder.Settings):
        class Task(str, enum.Enum):
            RETRIEVAL_QUERY: str = "retrieval.query"
            RETRIEVAL_PASSAGE: str = "retrieval.passage"
            SEPARATION: str = "separation"
            CLASSIFICATION: str = "classification"
            TEXT_MATCHING: str = "text-matching"

        normalize: bool = True
        task: Task = Task.RETRIEVAL_QUERY

    def __init__(self):
        """Initialize the embedder."""
        logger.info(f"Initializing Jina embeddings v3 embedder with model {self.MODEL_NAME}")
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.MODEL_REVISION
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.MODEL_REVISION
        ).to(self.DEVICE)

    # noinspection DuplicatedCode
    def batch_embed(self, texts: list[str], settings: Settings = None) -> list[DenseVector]:
        """Get embeddings for a batch of texts"""
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")

        if settings is None:
            settings = self.Settings()

        # Tokenize and prepare for model
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.DEVICE)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs, task=settings.task.value)

        # Apply mean pooling and optionally normalize
        token_embeddings = model_output[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        if settings.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return embeddings.cpu().numpy().tolist()
