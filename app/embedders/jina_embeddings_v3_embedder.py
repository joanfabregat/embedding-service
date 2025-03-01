# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import torch
from transformers import AutoTokenizer, AutoModel

from app.logging import logger
from app.utils import get_device
from .base_embedder import BaseEmbedder


class JinaEmbeddingsV3Embedder(BaseEmbedder):
    """
    Embedder using the Jina embeddings v3 model
    https://huggingface.co/jinaai/jina-embeddings-v3
    """

    MODEL_NAME = "jinaai/jina-embeddings-v3"
    REVISION = "f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
    DEFAULT_NORMALIZE = True
    DEFAULT_TASK = "retrieval.query"

    def __init__(self):
        """
        Initialize the embedder.
        """
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
    def batch_embed(self, texts: list[str], config: dict) -> list[list[float]]:
        """
        Get embeddings for a batch of texts

        Supported tasks are:
        - retrieval.query
        - retrieval.passage
        - separation
        - classification
        - text-matching

        Args:
            texts: The texts to get embeddings for
            config: The configuration for the model (supports 'task' and 'normalize')

        Returns:
            list[list[float]]: The embeddings for the texts
        """
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")

        # Tokenize and prepare for model
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs, task=config.get("task", self.DEFAULT_TASK))

        # Apply mean pooling and optionally normalize
        token_embeddings = model_output[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        if config.get("normalize", self.DEFAULT_NORMALIZE):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return embeddings.cpu().numpy().tolist()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens in

        Returns:
            int: The number of tokens in the text
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
