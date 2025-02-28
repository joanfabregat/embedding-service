#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information


import torch
from transformers import AutoTokenizer, AutoModel

from app.logging import logger
from app.utils import get_device
from .base_embedder import BaseTransformerEmbedder


class E5LargeV2Embedder(BaseTransformerEmbedder):
    """
    Embedder using the Multilingual E5 model
    """

    MODEL_NAME = "intfloat/e5-large-v2"

    def __init__(self):
        """
        Initialize the embedder.
        """
        logger.info(f"Initializing E5 embedder with model {self.MODEL_NAME}")
        self.device = get_device()
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(self.MODEL_NAME),
            model=AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        )

    def batch_embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Embed a batch of texts using the Multilingual E5 model.

        Args:
            texts: The texts to embed
            **kwargs: Additional arguments to pass to the embedder (e.g. normalize)

        Returns:
            list[float]: The embeddings of the texts
        """
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")

        prepared_texts = []
        for text in texts:
            if not text.startswith(("query:", "passage:")):
                prepared_texts.append(f"passage: {text}")
            else:
                prepared_texts.append(text)

        # Tokenize and prepare for model
        inputs = self.tokenizer(prepared_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)

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
