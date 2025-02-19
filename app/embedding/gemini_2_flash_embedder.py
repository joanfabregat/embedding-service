#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information


from app.types import DenseVector
from .base_embedder import BaseEmbedder


class Gemini2FlashEmbedder(BaseEmbedder):
    """
    An embedder that uses the Google GenAI API to create embeddings.
    """
    model_name = "gemini/text-embedding-005"
    is_dense = True

    def batch_embed(self, texts: list[str], batch_size: int = 32) -> list[DenseVector]:
        """
        Embed a batch of texts.

        Args:
            texts: The texts to embed.
            batch_size: The number of texts to embed in each batch. TODO

        Returns:
            The embeddings of the texts.
        """
        from app.services.google.genai import genai_client
        response = genai_client.models.embed_content(
            model="text-embedding-005",
            contents=texts
        )
        return [embedding.values for embedding in response.embeddings]
