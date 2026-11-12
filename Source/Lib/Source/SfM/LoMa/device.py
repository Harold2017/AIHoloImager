# Copyright (c) 2026 Minmin Gong
#
# Adapted from https://github.com/davnords/LoMa/blob/main/src/loma/device.py

import torch

try:
    from PythonSystem import ComputeDevice

    device = ComputeDevice()
except ImportError:
    # Standalone mode, e.g. running the benchmark harness outside of the application.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
else:
    amp_dtype = torch.bfloat16
