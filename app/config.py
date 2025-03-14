# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

import os

VERSION = os.getenv("VERSION") or "v0.0"
BUILD_ID = os.getenv("BUILD_ID") or "000000"
COMMIT_SHA = os.getenv("COMMIT_SHA") or "000000"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
