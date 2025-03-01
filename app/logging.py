#
#  @file download_models.py
#  @copyright Copyright (c) 2025 Fog&Frog
#  @author Joan Fabrégat <j@fabreg.at>
#  @license MIT
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import logging

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Add console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
