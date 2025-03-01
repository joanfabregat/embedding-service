#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import functools

from app.config import Config
from .base_embedder import BaseEmbedder


@functools.lru_cache(maxsize=None)
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


def get_embedder(model_nane: str) -> type[BaseEmbedder]:
    """
    Get the embedder for the given name

    Args:
        model_nane: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    if model_nane not in Config.ENABLED_MODELS:
        raise ValueError(f"Embedder {model_nane} is not enabled")

    match model_nane:
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
            raise ValueError(f"Embedder {model_nane} is not supported")
