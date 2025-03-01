# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from datetime import datetime

from app.config import Config
from app.logging import logger
from app.embedders import load_embedder

if __name__ == "__main__":
    logger.info(f"💽 Downloading model {Config.EMBEDDING_MODEL}")
    start = datetime.now()
    embedder = load_embedder(Config.EMBEDDING_MODEL)
    embedder.batch_embed(texts=["Hello, World!"])
    logger.info(f"✅ Finished downloading model {Config.EMBEDDING_MODEL} in {(datetime.now() - start).total_seconds()}s")
