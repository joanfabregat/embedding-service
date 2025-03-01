#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#


from datetime import datetime

from app.config import Config
from app.embedders import get_embedder
from app.logging import logger


def download_models():
    """
    Download the models
    """
    logger.info(f"ℹ️ Downloading {len(Config.ENABLED_MODELS)} enabled models: {Config.ENABLED_MODELS}")
    for model in Config.ENABLED_MODELS:
        logger.info(f"💽 Downloading model {model}")
        start = datetime.now()
        get_embedder(model)
        logger.info(f"✅ Finished downloading model {model} in {(datetime.now() - start).total_seconds()}s")


if __name__ == "__main__":
    download_models()
