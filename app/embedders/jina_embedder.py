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

        task: Task = Task.RETRIEVAL_QUERY
        batch_size: int = 4

    def __init__(self):
        """Initialize the embedder."""
        logger.info(f"Initializing Jina embeddings v3 embedder with model {self.MODEL_NAME}")
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.MODEL_REVISION
        )
        # Initialize model on CPU first for better memory management
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.MODEL_REVISION,
            low_cpu_mem_usage=True
        )

    def batch_embed(self, texts: list[str], settings: Settings = None) -> list[DenseVector]:
        """Get embeddings for a batch of texts with memory efficiency"""
        if not texts:
            return []

        if settings is None:
            settings = self.Settings()

        total_texts = len(texts)
        logger.info(f"Embedding {total_texts} texts using {self.MODEL_NAME} with batching")

        # Use batch size from settings
        batch_size = settings.batch_size
        all_embeddings: list[DenseVector] = []

        # Process texts in batches
        try:
            self._move_model_to_device()
            for i, batch in enumerate(self._create_batches(texts, batch_size)):
                try:
                    logger.debug(
                        f"Processing batch {i + 1}/{(total_texts + batch_size - 1) // batch_size} with {len(batch)} texts")

                    with torch.no_grad():
                        # Process a single batch
                        batch_output = self.model.encode(sentences=batch, task=settings.task.value)

                    # If output is a tensor, convert to list
                    if isinstance(batch_output, torch.Tensor):
                        batch_output = batch_output.detach().cpu().numpy().tolist()

                    # Handle different output types from the model's encode method
                    if isinstance(batch_output, list):
                        # Check if we need to do conversion from tensors
                        if batch_output and isinstance(batch_output[0], torch.Tensor):
                            batch_output = [t.cpu().numpy().tolist() for t in batch_output]

                    # Add batch results to the full results list
                    all_embeddings.extend(batch_output)

                    # Clean up batch resources
                    del batch_output
                finally:
                    # Move model back to CPU to free GPU memory
                    self._force_gc()
        finally:
            self._move_model_to_cpu()
            self._force_gc()

            logger.info(f"Completed embedding {total_texts} texts")
        return all_embeddings
