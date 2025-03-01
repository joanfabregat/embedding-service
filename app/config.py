#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import os


class Config:
    PORT = int(os.getenv("PORT", 8000))

    # Application configuration
    APP_VERSION = os.getenv("APP_VERSION", "v0.0")
    APP_BUILD_ID = os.getenv("APP_BUILD_ID", "XXXXXX")
    APP_COMMIT_SHA = os.getenv("APP_COMMIT_SHA", "XXXXXX")

    # Models
    ENABLED_MODELS = os.getenv("ENABLED_MODELS", "bm42,jina_embeddings_v3,e5_large_v2").split(",")
