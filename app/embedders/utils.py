#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

import functools
import torch


@functools.lru_cache()
def get_computation_device() -> torch.device:
    """Get the device to use for computation"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
