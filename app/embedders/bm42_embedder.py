# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import numpy as np
from fastembed import SparseTextEmbedding

from app.logging import logger
from .base_embedder import BaseEmbedder, SparseVector


class BM42Embedder(BaseEmbedder):
    """
    A sparse embedder for BM42 that returns sparse embeddings as (indices, values) pairs.
    https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions
    https://qdrant.tech/articles/bm42/
    
    Supports sliding window approach for embedding texts of any length.
    """

    MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    DEFAULT_SPARSITY_THRESHOLD = 0.005
    DEFAULT_ALLOW_NULL_VECTOR = False
    # Define a maximum token limit for the model
    MAX_TOKEN_LIMIT = 512
    # Define a default overlap for the sliding window (in tokens)
    DEFAULT_WINDOW_OVERLAP = 100
    EMBEDDING_TYPE = SparseVector
    DEVICE = "cpu"

    class BatchEmbedRequest(BaseEmbedder.BatchEmbedRequest):
        sparsity_threshold: float = 0.005
        allow_null_vector: bool = False
        window_size: int = 512
        window_overlap: int = 100
        window_combine_strategy: str = 'max'

    class BatchEmbedResponse(BaseEmbedder.BatchEmbedResponse):
        embeddings: list[SparseVector | None]

    def __init__(self):
        """
        Initialize the BM42 sparse embedder.
        """
        logger.info(f"Initializing BM42 sparse embedder with model {self.MODEL_NAME}")
        self.model = SparseTextEmbedding(model_name=self.MODEL_NAME)

    def batch_embed(self, request: BatchEmbedRequest) -> BatchEmbedResponse:
        """
        Embed a batch of texts into sparse vectors using sliding window approach for long texts.
        """
        logger.info(f"Embedding {len(request.texts)} texts using {self.MODEL_NAME}")

        # Categorize texts based on token count
        short_texts = []  # Texts that fit within window_size
        long_texts_info = []  # Tuples of (original_index, windows) for long texts

        for idx, text in enumerate(request.texts):
            if self._count_string_tokens(text) <= request.window_size:
                short_texts.append(text)
            else:
                windows = self._split_into_windows(text, request.window_size, request.window_overlap)
                long_texts_info.append((idx, windows))

        # Prepare result containers
        result_vectors: list[SparseVector | None] = [None] * len(request.texts)

        # Process short texts in a batch (if any)
        if short_texts:
            # Create a mapping of original indices
            short_text_map = []
            for idx, text in enumerate(request.texts):
                if self._count_string_tokens(text) <= request.window_size:
                    short_text_map.append(idx)

            # Embed all short texts and process them as a stream
            short_embeddings = self.model.embed(short_texts)
            for embedding, orig_idx in zip(short_embeddings, short_text_map):
                sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                if request.sparsity_threshold:
                    sparse_vector = self._apply_sparse_threshold(
                        sparse_vector,
                        request.sparsity_threshold,
                        request.allow_null_vector
                    )
                result_vectors[orig_idx] = sparse_vector

        # Process all windows from long texts as a single batch (if any)
        if long_texts_info:
            # Flatten all windows into a single batch
            all_windows = []
            window_map = []  # Maps each window back to its original text

            for orig_idx, windows in long_texts_info:
                for window in windows:
                    all_windows.append(window)
                    window_map.append(orig_idx)

            # Embed all windows in a single batch and process them as a stream
            all_window_embeddings = self.model.embed(all_windows)

            # Group window embeddings by original text
            text_to_windows = {}
            for embedding, orig_idx in zip(all_window_embeddings, window_map):
                if orig_idx not in text_to_windows:
                    text_to_windows[orig_idx] = []

                sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                if request.sparsity_threshold:
                    sparse_vector = self._apply_sparse_threshold(
                        sparse_vector,
                        request.sparsity_threshold,
                        request.allow_null_vector
                    )

                if sparse_vector:  # Only add non-None vectors
                    text_to_windows[orig_idx].append(sparse_vector)

            # Combine windows for each long text
            for orig_idx, window_vectors in text_to_windows.items():
                if window_vectors:
                    combined_vector = self._combine_sparse_vectors(window_vectors, request.combine_strategy)
                    result_vectors[orig_idx] = combined_vector
                else:
                    result_vectors[orig_idx] = None if request.allow_null_vector else ([], [])

        return self.BatchEmbedResponse(
            model=self.MODEL_NAME,
            embeddings=result_vectors,
        )

    def _split_into_windows(self, text: str, window_size: int, window_overlap: int) -> list[str]:
        """
        Split a text into overlapping windows based on token count.

        Args:
            text: The text to split
            window_size: Maximum size of each window in tokens
            window_overlap: Overlap between windows in tokens

        Returns:
            List of text windows
        """
        try:
            # Use the model's tokenizer for accurate splitting
            tokenizer = self.model.tokenizer
            tokens = tokenizer.encode(text)

            # If text fits in a single window, return it directly
            if len(tokens) <= window_size:
                return [text]

            # Create overlapping windows of tokens
            step_size = window_size - window_overlap
            window_indices = [(i, min(i + window_size, len(tokens)))
                              for i in range(0, len(tokens), step_size)]

            # Convert token indices back to text
            windows = []
            for start, end in window_indices:
                window_tokens = tokens[start:end]
                window_text = tokenizer.decode(window_tokens)
                windows.append(window_text)

            logger.info(f"Split text into {len(windows)} windows (token count: {len(tokens)})")
            return windows

        except (AttributeError, TypeError):
            # Fallback to a simpler approach using rough token estimation
            import re
            words = re.findall(r'\w+|[^\w\s]', text)

            # Estimate tokens per word (typically around 1.3)
            tokens_per_word = 1.3
            estimated_window_size = int(window_size / tokens_per_word)
            estimated_overlap = int(window_overlap / tokens_per_word)

            # If text fits in a single window, return it directly
            if len(words) <= estimated_window_size:
                return [text]

            # Create overlapping windows
            windows = []
            step_size = estimated_window_size - estimated_overlap

            for i in range(0, len(words), step_size):
                window_words = words[i:i + estimated_window_size]
                window_text = ' '.join(window_words)
                windows.append(window_text)

            logger.info(f"Split text into {len(windows)} windows using word-based estimation")
            return windows

    @staticmethod
    def _combine_sparse_vectors(
            sparse_vectors: list[SparseVector],
            strategy: str = 'max'
    ) -> SparseVector:
        """
        Combine multiple sparse vectors into a single sparse vector.

        Args:
            sparse_vectors: List of sparse vectors to combine
            strategy: Strategy for combining overlapping indices
                - 'max': Take the maximum absolute value
                - 'mean': Take the average value
                - 'sum': Sum all values

        Returns:
            Combined sparse vector
        """
        if not sparse_vectors:
            return [], []

        if len(sparse_vectors) == 1:
            return sparse_vectors[0]

        # Collect all indices and their corresponding values
        all_indices = {}

        for indices, values in sparse_vectors:
            for idx, val in zip(indices, values):
                if idx not in all_indices:
                    all_indices[idx] = []
                all_indices[idx].append(val)

        # Combine values for each index according to the strategy
        combined_indices = []
        combined_values = []

        for idx, vals in all_indices.items():
            combined_indices.append(idx)

            if strategy == 'max':
                # Take value with maximum absolute magnitude
                max_abs_val_idx = np.argmax(np.abs(vals))
                combined_values.append(vals[max_abs_val_idx])
            elif strategy == 'mean':
                # Take average of all values
                combined_values.append(np.mean(vals))
            elif strategy == 'sum':
                # Sum all values
                combined_values.append(np.sum(vals))
            else:
                # Default to max
                max_abs_val_idx = np.argmax(np.abs(vals))
                combined_values.append(vals[max_abs_val_idx])

        # Ensure indices are sorted (optional, but can be helpful)
        sorted_idx = np.argsort(combined_indices)
        combined_indices = [combined_indices[i] for i in sorted_idx]
        combined_values = [combined_values[i] for i in sorted_idx]

        return combined_indices, combined_values

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
                filtered_indices = [indices[max_idx]]
                filtered_values = [values[max_idx]]

        return filtered_indices, filtered_values

    def count_tokens(self, request: BaseEmbedder.TokensCountRequest) -> BaseEmbedder.TokensCountResponse:
        """
        Count the number of tokens in a batch of texts.
        """
        logger.info(f"Counting tokens for {len(request.texts)} texts using {self.MODEL_NAME}")
        return self.TokensCountResponse(
            model=self.MODEL_NAME,
            tokens_count=[self._count_string_tokens(text) for text in request.texts],
        )

    def _count_string_tokens(self, string: str) -> int:
        """
        Count the number of tokens in a text string.
        """
        return len(self.model.tokenizer.encode(string))
