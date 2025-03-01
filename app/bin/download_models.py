# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

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
