# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from app.config import Config
from app.logging import logger
from .base_embedder import BaseEmbedder


def load_embedder(model_name: str = Config.EMBEDDING_MODEL) -> BaseEmbedder:
    """
    Load the embedder for the given name

    Args:
        model_name: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    model_name = get_embedder(model_name)
    return model_name()


def get_embedder(model_name: Config.EMBEDDING_MODEL) -> type[BaseEmbedder]:
    """
    Get the embedder for the given name

    Args:
        model_name: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    logger.info(f"Loading embedder for {model_name}")

    if model_name not in Config.ENABLED_MODELS:
        raise ValueError(f"Embedder {model_name} is not enabled")

    match model_name:
        case "bm42":
            from .bm42_embedder import BM42Embedder
            return BM42Embedder

        case "jina_embeddings_v3":
            from .jina_embeddings_v3_embedder import JinaEmbeddingsV3Embedder
            return JinaEmbeddingsV3Embedder

        case "e5_large_v2":
            from .e5_large_v2_embedder import E5LargeV2Embedder
            return E5LargeV2Embedder

        case _:
            raise ValueError(f"Embedder {model_name} is not supported")
