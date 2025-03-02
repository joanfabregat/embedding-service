# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from app.logging import logger
from .base_embedder import BaseEmbedder


def load_embedder(model_name: str) -> BaseEmbedder:
    """
    Load the embedder for the given name

    Args:
        model_name: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    model_name = get_embedder(model_name)
    return model_name()


def get_embedder(model_name: str) -> type[BaseEmbedder]:
    """
    Get the embedder for the given name

    Args:
        model_name: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    logger.info(f"Loading embedder for {model_name}")

    match model_name:
        case "bm42":
            from .bm42_embedder import BM42Embedder
            return BM42Embedder

        case "jina":
            from .jina_embedder import JinaEmbedder
            return JinaEmbedder

        case "e5":
            from .e5_embedder import E5Embedder
            return E5Embedder

        case _:
            raise ValueError(f"Embedder {model_name} is not supported")
