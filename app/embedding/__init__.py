#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from .base_embedder import BaseEmbedder
from .bge_m3_embedder import BgeM3Embedder
from .bm25_embedder import Bm25Embedder
from .gemini_2_flash_embedder import Gemini2FlashEmbedder
from .multilingual_e5_large_embedder import MultilingualE5LargeEmbedder
from .splade_cocondenser_embedder import SpladeCocondenserEmbedder
from .st_mpnet_v2_embedder import StMpnetV2Embedder

EMBEDDERS = [
    BgeM3Embedder,
    Bm25Embedder,
    Gemini2FlashEmbedder,
    MultilingualE5LargeEmbedder,
    SpladeCocondenserEmbedder,
    StMpnetV2Embedder
]


def load_embedder(model_name: str, allow_gpu: bool = False) -> BaseEmbedder:
    """
    Load an embedder based on the model name

    Args:
        model_name: The model name to load
        allow_gpu: Whether to allow GPU usage

    Returns:
        An instance of an embedder
    """
    model = next((embedder for embedder in EMBEDDERS if embedder.model_name == model_name), None)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    return model(allow_gpu=allow_gpu)
