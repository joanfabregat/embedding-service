#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import logging

from app.config import APP_ENV, __PROJECT_ROOT__


def create_file_handler(filename: str, level: int = logging.DEBUG) -> logging.FileHandler:
    """
    Create a file handler for logging

    Args:
        filename: The name of the file to log to
        level: The logging level

    Returns:
        The file handler
    """
    file_handler = logging.FileHandler(__PROJECT_ROOT__ / "var" / "logs" / filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    return file_handler


def create_console_handler(level: int = logging.DEBUG) -> logging.StreamHandler:
    """
    Create a console handler for logging

    Args:
        level: The logging level

    Returns:
        The console handler
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    return console_handler


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Logging to file in development
if APP_ENV == "development":
    logger.addHandler(create_file_handler(filename="app.log"))

# Logging to console
logger.addHandler(create_console_handler(level=logging.INFO))
