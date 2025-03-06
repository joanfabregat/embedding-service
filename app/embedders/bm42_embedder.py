#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

import re
import gc
import numpy as np
from fastembed import SparseTextEmbedding
import enum
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

    class Settings(BaseEmbedder.Settings):
        class Task(str, enum.Enum):
            QUERY: str = "query"
            INDEX: str = "index"

        task: Task = Task.QUERY
        sparsity_threshold: float = 0.005
        allow_null_vector: bool = False
        window_size: int = 512  # Define a maximum token limit for the model
        window_overlap: int = 100  # Define a default overlap for the sliding window (in tokens)
        window_combine_strategy: str = 'max'

    def __init__(self):
        """
        Initialize the BM42 sparse embedder.
        """
        logger.info(f"Initializing BM42 sparse embedder with model {self.MODEL_NAME}")
        self.model = SparseTextEmbedding(model_name=self.MODEL_NAME)

    def batch_embed(self, texts: str, settings: Settings = None) -> list[SparseVector | None]:
        """
        Embed a batch of texts into sparse vectors using sliding window approach for long texts.
        """
        texts_count = len(texts)
        logger.info(f"Embedding {texts_count} texts using {self.MODEL_NAME}")

        # Prepare result containers
        result_vectors: list[SparseVector | None] = [None] * texts_count

        try:
            if settings is None:
                settings = self.Settings()

            # Categorize texts based on token count
            short_texts = []  # Texts that fit within window_size
            long_texts_info = []  # Tuples of (original_index, windows) for long texts

            for idx, text in enumerate(texts):
                if self.count_tokens(text) <= settings.window_size:
                    short_texts.append(text)
                else:
                    windows = self._split_into_windows(text, settings.window_size, settings.window_overlap)
                    long_texts_info.append((idx, windows))

            # Process short texts in a batch (if any)
            if short_texts:
                # Create a mapping of original indices
                short_text_map = []
                for idx, text in enumerate(texts):
                    if self.count_tokens(text) <= settings.window_size:
                        short_text_map.append(idx)

                # Embed all short texts and process them as a stream
                short_embeddings = self._embed(short_texts, settings)
                for embedding, orig_idx in zip(short_embeddings, short_text_map):
                    sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                    if settings.sparsity_threshold:
                        sparse_vector = self._apply_sparse_threshold(
                            sparse_vector,
                            settings.sparsity_threshold,
                            settings.allow_null_vector
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
                all_window_embeddings = self._embed(all_windows, settings)

                # Group window embeddings by original text
                text_to_windows = {}
                for embedding, orig_idx in zip(all_window_embeddings, window_map):
                    if orig_idx not in text_to_windows:
                        text_to_windows[orig_idx] = []

                    sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                    if settings.sparsity_threshold:
                        sparse_vector = self._apply_sparse_threshold(
                            sparse_vector,
                            settings.sparsity_threshold,
                            settings.allow_null_vector
                        )

                    if sparse_vector:  # Only add non-None vectors
                        text_to_windows[orig_idx].append(sparse_vector)

                # Combine windows for each long text
                for orig_idx, window_vectors in text_to_windows.items():
                    if window_vectors:
                        combined_vector = self._combine_sparse_vectors(window_vectors, settings.window_combine_strategy)
                        result_vectors[orig_idx] = combined_vector
                    else:
                        result_vectors[orig_idx] = None if settings.allow_null_vector else ([], [])
        finally:
            gc.collect()

        return result_vectors

    def _embed(self, texts: list[str], settings: Settings):
        """Embed a list of texts using the BM42 model."""
        if settings.task == self.Settings.Task.QUERY:
            return self.model.query_embed(texts)
        elif settings.task == self.Settings.Task.INDEX:
            return self.model.embed(texts)
        else:
            raise ValueError(f"Unsupported task: {settings.task}")

    @staticmethod
    def _split_into_windows(text: str, window_size: int, window_overlap: int) -> list[str]:
        """
        Split a text into overlapping windows based on token count.

        Args:
            text: The text to split
            window_size: Maximum size of each window in tokens
            window_overlap: Overlap between windows in tokens

        Returns:
            List of text windows
        """
        # Use a simple word-based approach with estimated token counts
        # Most tokenizers treat words as roughly 1.3 tokens on average
        tokens_per_word = 1.3
        words = re.findall(r'\w+|[^\w\s]', text)

        # Estimate the word counts based on token limits
        estimated_window_size = int(window_size / tokens_per_word)
        estimated_overlap = int(window_overlap / tokens_per_word)

        # If text fits in a single window, return it directly
        if len(words) <= estimated_window_size:
            return [text]

        # Create overlapping windows
        windows = []
        step_size = estimated_window_size - estimated_overlap

        for i in range(0, len(words), max(1, int(step_size))):
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

        # Check if values is empty before proceeding
        if not values:
            return ([], []) if not allow_null_vector else None

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
                # Only try to find max if values is not empty
                if values:
                    max_idx = np.argmax(np.abs(values))
                    filtered_indices = [indices[max_idx]]
                    filtered_values = [values[max_idx]]
                else:
                    # Return empty lists if values is empty
                    return [], []

        return filtered_indices, filtered_values

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        """
        # Use a simple estimation based on word count
        # Most tokenizers treat words as roughly 1.3 tokens on average
        words = re.findall(r'\w+|[^\w\s]', text)
        return int(len(words) * 1.3)
