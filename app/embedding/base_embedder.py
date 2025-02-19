#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from app.types import SparseVector, DenseVector


class BaseEmbedder:
    """
    Base class for embedding text
    """

    model_name: str = ...
    is_sparse: bool = False
    is_dense: bool = False

    def batch_embed(self, texts: list[str], batch_size: int = 32) -> list[SparseVector | DenseVector]:
        """
        Embeds a batch of texts

        Args:
            texts: list of texts to embed
            batch_size: size of the

        Returns:
            list of embeddings
        """
        ...
