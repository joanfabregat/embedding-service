import enum
import gc

import torch
from transformers import AutoTokenizer, AutoModel

from app.logging import logger
from app.models import DenseVector
from .base_transformer_embedder import BaseTransformerEmbedder


class E5Embedder(BaseTransformerEmbedder):
    """
    Embedder using the Multilingual E5 model
    https://huggingface.co/intfloat/e5-large-v2
    """

    MODEL_NAME = "intfloat/e5-large-v2"
    SUPPORTED_PREFIXES = ("query:", "passage:")

    class Settings(BaseTransformerEmbedder.Settings):
        class Task(str, enum.Enum):
            QUERY: str = "query"
            INDEX: str = "index"

        task: Task = Task.QUERY
        batch_size: int = 25

    def __init__(self):
        """Initialize the embedder."""
        super().__init__()
        logger.info(f"Initializing E5 embedder with model {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            low_cpu_mem_usage=True  # Use memory-efficient initialization
        )

    # noinspection DuplicatedCode
    def batch_embed(self, texts: list[str], settings: Settings = None) -> list[DenseVector]:
        """Embed a batch of texts using the Multilingual E5 model with memory optimization."""
        if not texts:
            return []

        if settings is None:
            settings = self.Settings()

        total_texts = len(texts)
        logger.info(f"Embedding {total_texts} texts using {self.MODEL_NAME} with batching")

        # Prepare texts with proper prefixes
        prepared_texts = []
        for text in texts:
            if not text.startswith(self.SUPPORTED_PREFIXES):
                prepared_texts.append(f"{settings.task.value}: {text}")
            else:
                prepared_texts.append(text)

        # Use batch size from settings
        all_embeddings: list[DenseVector] = []

        try:
            self._move_model_to_device()

            # Process in smaller batches to avoid memory issues
            for i, batch in enumerate(self._create_batches(prepared_texts, settings.batch_size)):
                logger.debug(f"Processing batch "
                             f"{i + 1}/{(total_texts + settings.batch_size - 1) // settings.batch_size} "
                             f"with {len(batch)} texts")

                # Tokenize and prepare batch for model
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.DEVICE)

                # Generate embeddings for this batch
                with torch.no_grad():
                    model_output = self.model(**inputs)

                # Apply mean pooling and optionally normalize
                token_embeddings = model_output[0]
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = (
                        torch.sum(token_embeddings * input_mask_expanded, 1) /
                        torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                )

                if settings.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to numpy and add to results
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

                # Clean up batch resources
                del inputs, model_output, token_embeddings, input_mask_expanded, embeddings, batch_embeddings
                self._force_gc()

            logger.info(f"Completed embedding {total_texts} texts")
            return all_embeddings

        finally:
            # Move model back to CPU to free GPU memory
            self._move_model_to_cpu()
            self._force_gc()
