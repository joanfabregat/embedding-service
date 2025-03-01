# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import os

VERSION = os.getenv("APP_VERSION", "v0.0")
BUILD_ID = os.getenv("APP_BUILD_ID", "000000")
COMMIT_SHA = os.getenv("APP_COMMIT_SHA", "000000")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
