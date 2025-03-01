#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from datetime import datetime

from app.config import Config
from app.embedders import get_embedder
from app.logging import logger


def download_models():
    """
    Download the models
    """
    logger.info(f"ℹ️ Downloading {len(Config.ENABLED_MODELS)} enabled models")
    for model in Config.ENABLED_MODELS:
        logger.info(f"💽 Downloading model {model}")
        start = datetime.now()
        get_embedder(model)
        logger.info(f"✅ Finished downloading model {model} in {(datetime.now() - start).total_seconds()}s")


if __name__ == "__main__":
    download_models()
