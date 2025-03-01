# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import torch
from transformers import AutoTokenizer, AutoModel

from app.logging import logger
from .base_dense_embedder import BaseDenseEmbedder, DenseVector


class E5LargeV2Embedder(BaseDenseEmbedder):
    """
    Embedder using the Multilingual E5 model
    https://huggingface.co/intfloat/e5-large-v2
    """

    DEFAULT_NORMALIZE = True
    MODEL_NAME = "intfloat/e5-large-v2"
    EMBEDDING_TYPE = DenseVector

    def __init__(self):
        """Initialize the embedder."""
        super().__init__()
        logger.info(f"Initializing E5 embedder with model {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.DEVICE)

    # noinspection DuplicatedCode
    def batch_embed(self, request: BaseDenseEmbedder.BatchEmbedRequest) -> BaseDenseEmbedder.BatchEmbedResponse:
        """Embed a batch of texts using the Multilingual E5 model."""
        logger.info(f"Embedding {len(request.texts)} texts using {self.MODEL_NAME}")

        prepared_texts = []
        for text in request.texts:
            if not text.startswith(("query:", "passage:")):
                prepared_texts.append(f"passage: {text}")
            else:
                prepared_texts.append(text)

        # Tokenize and prepare for model
        inputs = self.tokenizer(prepared_texts, padding=True, truncation=True, return_tensors="pt").to(self.DEVICE)

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

        if request.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return self.BatchEmbedResponse(
            model=self.MODEL_NAME,
            embeddings=embeddings.cpu().numpy().tolist(),
        )
