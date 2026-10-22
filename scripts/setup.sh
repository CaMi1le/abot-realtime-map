#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor models data artifacts
python3 -m venv --without-pip .venv
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -fL https://bootstrap.pypa.io/get-pip.py -o artifacts/get-pip.py
  .venv/bin/python artifacts/get-pip.py
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if [ ! -d vendor/ABot-Recon/.git ]; then
  git clone https://github.com/amap-cvlab/ABot-Recon.git vendor/ABot-Recon
fi
git -C vendor/ABot-Recon checkout 6e3b2e1257464a74dfceff22eb33190bd228276e
.venv/bin/pip install -e 'vendor/ABot-Recon[loop]' -e . 'huggingface-hub<1' ninja
.venv/bin/pip freeze > artifacts/requirements-lock.txt
if [ "${REALTIME_MAP_SKIP_DOWNLOAD:-0}" != 1 ]; then
  .venv/bin/python scripts/download.py
  .venv/bin/python vendor/ABot-Recon/scripts/download_loop_assets.py --output-dir models/loop
fi
