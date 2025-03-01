#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information


class BaseEmbedder:
    """
    Base class for embedders
    """

    def batch_embed(self, texts: list[str], config: dict):
        """
        Embed a batch of texts.

        Args:
            texts: The texts to embed
            config: The configuration for the model

        Returns:
            list: The embeddings of the texts
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens in

        Returns:
            int: The number of tokens in the text
        """
        raise NotImplementedError
