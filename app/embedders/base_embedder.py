# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

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
