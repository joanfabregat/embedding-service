#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from .base_transformer_dense_embbeder import BaseTransformerDenseEmbedder


class StMpnetV2Embedder(BaseTransformerDenseEmbedder):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self, allow_gpu: bool = False):
        super().__init__(model_name=self.model_name, allow_gpu=allow_gpu)
