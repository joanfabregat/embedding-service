#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import enum

from transformers import AutoTokenizer, AutoModel
import torch

from app.logging import logger
from app.utils import get_device
from .base_embedder import BaseTransformerEmbedder


class JinaEmbeddingsV3Embedder(BaseTransformerEmbedder):
    """
    Embedder using the Jina embeddings v3 model
    """

    class Tasks(str, enum.Enum):
        """The tasks that the model can be used"""
        RETRIEVAL_QUERY = "retrieval.query"
        RETRIEVAL_PASSAGE = "retrieval.passage"
        SEPARATION = "separation"
        CLASSIFICATION = "classification"
        TEXT_MATCHING = "text-matching"

    MODEL_NAME = "jinaai/jina-embeddings-v3"

    def __init__(self):
        """
        Initialize the embedder.
        """
        logger.info(f"Initializing Jina embeddings v3 embedder with model {self.MODEL_NAME}")
        self.device = get_device()
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True),
            model=AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.device)
        )

    # Process batch of texts to embeddings
    def batch_embed(
            self,
            texts: list[str],
            **kwargs,
    ) -> list[list[float]]:
        """
        Get embeddings for a batch of texts

        Args:
            texts: The texts to get embeddings for
            **kwargs: Additional arguments to pass to the embedder (e.g. normalize, task)

        Returns:
            list[list[float]]: The embeddings for the texts
        """
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")

        # Tokenize and prepare for model
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs, task=kwargs.get("task", self.Tasks.RETRIEVAL_QUERY))

        # Apply mean pooling and optionally normalize
        token_embeddings = model_output[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        if kwargs.get("normalize", True):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return embeddings.cpu().numpy().tolist()
