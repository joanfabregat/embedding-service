#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import os

class Config:
    # Application configuration
    APP_VERSION = os.getenv("APP_VERSION", "unknown")
    APP_BUILD_ID = os.getenv("APP_BUILD_ID", "unknown")
    APP_COMMIT_SHA = os.getenv("APP_COMMIT_SHA", "unknown")

    # Models
    ENABLE_E5_LARGE_V2 = os.getenv("ENABLE_E5_LARGE_V2", "false").lower() == "true"
    ENABLE_JINA_EMBEDDINGS_V3 = os.getenv("ENABLE_JINA_EMBEDDINGS_V3", "false").lower() == "true"
    ENABLE_BM42 = os.getenv("ENABLE_BM42", "false").lower() == "true"
