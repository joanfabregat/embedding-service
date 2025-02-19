#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import gc

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel

from app.logger import logger
from app.types import SparseVector
from .base_transformer_embbeder import BaseTransformerEmbedder


class BaseTransformerSparseEmbedder(BaseTransformerEmbedder):
    """
    A class to embed text using a Transformer sparse model.
    """

    is_sparse = True

    def __init__(
            self,
            model_name: str,
            allow_gpu: bool = False,
            trust_remote_code: bool = False,
            masked_lm: bool = True
    ):
        self.allow_gpu = allow_gpu
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.masked_lm = masked_lm
        logger.info(f"Loading sparse model: {self.model_name}")
        model_loader = AutoModel if not masked_lm else AutoModelForMaskedLM
        self.model = model_loader.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_safetensors=True
        )
        self.model.to(self.get_device(self.allow_gpu))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Loaded dense model: {self.model_name}")

    def batch_embed(self, texts: list[str], batch_size: int = 32) -> list[SparseVector]:
        """
        Embeds a list of texts using the model and tokenizer.

        Args:
            texts: A list of texts to embed.
            batch_size: The number of texts to embed in each batch.

        Returns:
            list[torch.Tensor]: A list of vector representations of the texts.
        """
        batches = range(0, len(texts), batch_size)
        batches_count = len(batches)
        logger.info(f"Embedding {len(texts)} texts with model '{self.model_name}' in {batches_count} batches")

        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before embedding")

        results = []
        for batch_num, i in enumerate(batches, 1):
            logger.info(f"Embedding batch {batch_num}/{batches_count}")
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)

                output = self.model(**tokens)
                attention_mask = tokens.attention_mask

                relu_log = torch.log(1 + torch.relu(output.logits))
                weighted_log = relu_log * attention_mask.unsqueeze(-1)
                max_val, _ = torch.max(weighted_log, dim=1)
                results.extend(max_val.cpu().tolist())

                del tokens, output, relu_log, weighted_log, max_val
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.mps.is_available():
                    torch.mps.empty_cache()

        vectors: list[SparseVector] = []
        try:
            for result in results:
                tensor = torch.tensor(result, device='cpu')
                sparse_tensor = tensor.to_sparse_coo().coalesce()
                indices = sparse_tensor.indices()[0].tolist()
                values = sparse_tensor.values().tolist()
                vectors.append((indices, values))
                del tensor, sparse_tensor
        finally:
            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return vectors
