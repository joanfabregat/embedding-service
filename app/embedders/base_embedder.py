#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseEmbedder:
    """
    Base class for embedders
    """

    def batch_embed(self, texts: list[str], **kwargs):
        """
        Embed a batch of texts.

        Args:
            texts: The texts to embed
            **kwargs: Additional arguments

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


class BaseTransformerEmbedder(BaseEmbedder):
    """
    Base class for embedders
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        """
        Initialize the embedder.

        Args:
            tokenizer: The tokenizer to use
            model: The model to use
        """
        self.tokenizer = tokenizer
        self.model = model

    def batch_embed(self, texts: list[str], **kwargs):
        """
        Embed a batch of texts.

        Args:
            texts: The texts to embed
            **kwargs: Additional arguments

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
        return len(self.tokenizer.encode(text, add_special_tokens=False))
