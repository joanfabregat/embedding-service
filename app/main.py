#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

from datetime import datetime

from fastapi import FastAPI

from app.config import Config
from app.logging import logger
from app.models import RootResponse
from app.utils import get_device

startup = datetime.now()

logger.info(f"Starting Embedding Service v{Config.APP_VERSION} ({Config.APP_BUILD_ID})")
app = FastAPI(
    title="Embedding Service",
    version=Config.APP_VERSION,
    description="API for generating sparse and dense embeddings from text"
)


@app.get("/", response_model=RootResponse, tags=["root"])
def root():
    return RootResponse(
        version=Config.APP_VERSION,
        build_id=Config.APP_BUILD_ID,
        commit_sha=Config.APP_COMMIT_SHA,
        uptime=(datetime.now() - startup).total_seconds(),
        device=get_device(),
    )


if Config.ENABLE_E5_LARGE_V2 or True:
    from app.e5_large_v2 import router as e5_large_v2_router

    app.include_router(e5_large_v2_router)

if Config.ENABLE_JINA_EMBEDDINGS_V3:
    from app.jina_embeddings_v3 import router as jina_embeddings_v3_router

    app.include_router(jina_embeddings_v3_router)
