#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import gc

import torch
from transformers import AutoModel, AutoTokenizer

from app.logger import logger
from app.types import DenseVector
from .base_transformer_embbeder import BaseTransformerEmbedder


class BaseTransformerDenseEmbedder(BaseTransformerEmbedder):
    """
    A dense embedder that uses a transformer model to create embeddings.
    """

    is_dense = True

    def __init__(self, model_name: str, allow_gpu: bool = bool, add_special_tokens: bool = True):
        self.allow_gpu = allow_gpu
        self.model_name = model_name
        self.add_special_tokens = add_special_tokens
        logger.info(f"Loading dense model: {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True)
        self.model.to(self.get_device(self.allow_gpu))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Loaded dense model: {self.model_name}")

    def batch_embed(self, texts: list[str], batch_size: int = 32) -> list[DenseVector]:
        """
        Create embeddings for the given texts in batches.

        Args:
            texts: The texts to embed.
            batch_size: Size of batches to process.

        Returns:
            list[DenseVector]: The embeddings of the texts.
        """
        batches = range(0, len(texts), batch_size)
        batches_count = len(batches)
        logger.info(f"Embedding {len(texts)} texts using dense model '{self.model_name}' in {batches_count} batches")

        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before embedding")

        tensors = []
        for batch_num, i in enumerate(batches, 1):
            logger.info(f"Embedding batch {batch_num}/{batches_count}")
            batch_texts = texts[i:i + batch_size]

            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens
            )
            tokens = tokens.to(self.model.device)

            with torch.no_grad():
                model_output = self.model(**tokens)
                batch_tensors = model_output.last_hidden_state.mean(dim=1).cpu()
                tensors.extend(batch_tensors)

            # Explicitly clear GPU memory
            del tokens, model_output
            torch.cuda.empty_cache()
            torch.mps.empty_cache()

        vectors: list[DenseVector] = []
        try:
            for tensor in tensors:
                if hasattr(tensor, 'cpu'):
                    tensor = tensor.cpu()
                vectors.append(tensor.tolist())
        finally:
            del tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return vectors
