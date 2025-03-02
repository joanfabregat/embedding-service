# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import enum
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

    class Settings(BaseTransformerEmbedder.Settings):
        class Task(str, enum.Enum):
            QUERY: str = "query"
            PASSAGE: str = "passage"

        task: Task = Task.QUERY

    def __init__(self):
        """Initialize the embedder."""
        super().__init__()
        logger.info(f"Initializing E5 embedder with model {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.DEVICE)

    # noinspection DuplicatedCode
    def batch_embed(self, texts: list[str], settings: Settings = None) -> list[DenseVector]:
        """Embed a batch of texts using the Multilingual E5 model."""
        logger.info(f"Embedding {len(texts)} texts using {self.MODEL_NAME}")

        if settings is None:
            settings = self.Settings()

        prepared_texts = []
        supported_prefixes = (f"{self.Settings.Task.QUERY.value}:", f"{self.Settings.Task.PASSAGE.value}:")
        for text in texts:
            if not text.startswith(supported_prefixes):
                prepared_texts.append(f"{settings.task.value}: {text}")
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

        if settings.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and then to list for JSON serialization
        return embeddings.cpu().numpy().tolist()
