#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from app.models import BatchEmbedRequest
from .embedding import JinaEmbeddingTasks


class JinaBatchEmbedRequest(BatchEmbedRequest):
    """Request schema for embeddings"""
    texts: list[str]
    task: JinaEmbeddingTasks
    normalize: bool = True
