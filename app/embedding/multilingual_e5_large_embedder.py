#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from .base_transformer_dense_embbeder import BaseTransformerDenseEmbedder

class MultilingualE5LargeEmbedder(BaseTransformerDenseEmbedder):
    model_name = "intfloat/multilingual-e5-large"

    def __init__(self, allow_gpu: bool = False):
        super().__init__(self.model_name, allow_gpu=allow_gpu, add_special_tokens=True)