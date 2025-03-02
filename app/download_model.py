# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

from datetime import datetime

from app.config import EMBEDDING_MODEL
from app.logging import logger
from app.embedders import load_embedder

if __name__ == "__main__":
    logger.info(f"💽 Downloading model {EMBEDDING_MODEL}")
    start = datetime.now()
    embedder = load_embedder(EMBEDDING_MODEL)
    embedder.batch_embed(texts=["hello", "world"])
    logger.info(f"✅ Finished downloading model {EMBEDDING_MODEL} in {(datetime.now() - start).total_seconds()}s")
