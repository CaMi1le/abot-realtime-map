#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export CC=gcc-12 CXX=g++-12 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
.venv/bin/pip install flashinfer-python==0.2.5 --no-deps
(cd vendor/ABot-Recon/abot_recon/modeling/pi3/models/curope && python setup.py build_ext --inplace)
python scripts/on-gpu.py python scripts/parity.py --attention paged
