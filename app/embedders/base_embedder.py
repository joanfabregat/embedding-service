#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

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
