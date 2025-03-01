#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import numpy as np
from fastembed import SparseTextEmbedding

from app.logging import logger
from app.models import SparseVector
from .base_embedder import BaseEmbedder


class BM42Embedder(BaseEmbedder):
    """
    A sparse embedder for BM42 that returns sparse embeddings as (indices, values) pairs.
    https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions
    https://qdrant.tech/articles/bm42/
    """

    MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    DEFAULT_SPARSITY_THRESHOLD = 0.005
    DEFAULT_ALLOW_NULL_VECTOR = False

    def __init__(self):
        """
        Initialize the BM42 sparse embedder.
        """
        logger.info(f"Initializing BM42 sparse embedder with model {self.MODEL_NAME}")
        self.model = SparseTextEmbedding(model_name=self.MODEL_NAME)

    def batch_embed(self, texts: list[str], config: dict) -> list[SparseVector or None]:
        """
        Embed a batch of texts into sparse vectors.

        Args:
            texts: A list of texts to embed
            config: A dictionary of configuration options (supports 'sparsity_threshold' and 'allow_null_vector')
            sparsity_threshold: The threshold for sparsity
            allow_null_vector: Whether to allow null vectors

        Returns:
            A list of sparse vectors
        """
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")
        embeddings = self.model.embed(texts)

        sparse_vectors = []
        for embedding in embeddings:
            sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
            sparsity_threshold = config.get('sparsity_threshold', self.DEFAULT_SPARSITY_THRESHOLD)
            allow_null_vector = config.get('allow_null_vector', self.DEFAULT_ALLOW_NULL_VECTOR)
            if sparsity_threshold:
                sparse_vector = self._apply_sparse_threshold(sparse_vector, sparsity_threshold, allow_null_vector)
            sparse_vectors.append(sparse_vector)

        return sparse_vectors

    @staticmethod
    def _apply_sparse_threshold(
            sparse_vector: SparseVector,
            sparsity_threshold: float,
            allow_null_vector: bool
    ) -> SparseVector or None:
        """
        Filter out values below the sparsity threshold.

        Args:
            sparse_vector: A sparse vector as (indices, values) pair
            sparsity_threshold: The threshold for sparsity
            allow_null_vector: Whether to allow null vectors

        Returns:
            A filtered sparse vector
        """
        indices, values = sparse_vector

        filtered_indices: list[int] = []
        filtered_values: list[float] = []
        for i, value in enumerate(values):
            if abs(value) >= sparsity_threshold:
                filtered_indices.append(indices[i])
                filtered_values.append(value)

        # If all values were filtered out, keep the highest magnitude value
        if not filtered_values:
            if allow_null_vector:
                return None
            else:
                max_idx = np.argmax(np.abs(values))
                filtered_values = [indices[max_idx]]
                filtered_indices = [values[max_idx]]

        return filtered_indices, filtered_values

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The input text string

        Returns:
            The number of tokens in the text
        """
        # Use the model's tokenizer to count tokens
        # SparseTextEmbedding doesn't directly expose the tokenizer count method
        # so we can either:
        # 1. Use the model's internal tokenizer
        # 2. Estimate based on a simple approach

        # Method 1: Using the model's tokenizer (preferred)
        try:
            # Access the internal tokenizer from the model
            tokenizer = self.model.tokenizer
            encoded = tokenizer.encode(text)
            return len(encoded)
        except (AttributeError, TypeError):
            # Fallback method if tokenizer is not directly accessible
            # Simple whitespace-based approximation (less accurate)
            import re
            # Split on whitespace and punctuation
            tokens = re.findall(r'\w+|[^\w\s]', text)
            return len(tokens)
