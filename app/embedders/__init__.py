#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import enum
import functools
from .base_embedder import BaseEmbedder
from .bm42_embedder import BM42Embedder
from .e5_large_v2_embedder import E5LargeV2Embedder
from .jina_embeddings_v3_embedder import JinaEmbeddingsV3Embedder


class AvailableEmbedders(str, enum.Enum):
    BM42 = "bm42"
    JINA_EMBEDDINGS_V3 = "jina_embeddings_v3"
    E5_LARGE_V2 = "e5_large_v2"


EMBEDDERS_MAPPING = {
    AvailableEmbedders.BM42: BM42Embedder,
    AvailableEmbedders.JINA_EMBEDDINGS_V3: JinaEmbeddingsV3Embedder,
    AvailableEmbedders.E5_LARGE_V2: E5LargeV2Embedder,
}


@functools.lru_cache(maxsize=None)
def load_embedder(embedder: AvailableEmbedders) -> BaseEmbedder:
    """
    Load the embedder for the given name

    Args:
        embedder: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    embedder = get_embedder(embedder)
    return embedder()


def get_embedder(embedder: AvailableEmbedders) -> type[BaseEmbedder]:
    """
    Get the embedder for the given name

    Args:
        embedder: The name of the embedder

    Returns:
        BaseEmbedder: The embedder
    """
    embedder = EMBEDDERS_MAPPING[embedder]
    if not embedder:
        raise ValueError(f"Embedder {embedder} not found")
    return embedder
