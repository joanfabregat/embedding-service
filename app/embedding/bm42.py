#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from functools import partial
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModel

from app.models import SparseVector


def normalize_sparse_vector(sparse_vector: SparseVector) -> SparseVector:
    """
    Normalize a sparse vector to unit length (L2 norm = 1).

    Args:
        sparse_vector: A tuple of (indices, values) representing a sparse vector

    Returns:
        A tuple of (indices, values) with the same indices but normalized values
    """
    indices, values = sparse_vector

    # Calculate the L2 norm (Euclidean norm)
    norm = sum(val ** 2 for val in values) ** 0.5

    # Avoid division by zero
    if norm < 1e-10:
        return indices, values

    # Normalize the values
    normalized_values = [val / norm for val in values]

    return indices, normalized_values

from fastembed import SparseTextEmbedding, TextEmbedding

query_text = "best programming language for beginners?"

model_bm42 = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

sparse_embedding = list(model_bm42.query_embed(query_text))[0]
dense_embedding = list(model_jina.query_embed(query_text))[0]

