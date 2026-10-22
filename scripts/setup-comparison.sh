#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
set -euo pipefail
cd "$(dirname "$0")/.."
export CC=gcc-12 CXX=g++-12 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
python3 -m venv --without-pip .venv-mast3r
SITE=$(.venv-mast3r/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$PWD/.venv/lib/python3.12/site-packages" > "$SITE/abot-environment.pth"
if [ ! -d vendor/MASt3R-SLAM/.git ]; then
  git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git vendor/MASt3R-SLAM
fi
git -C vendor/MASt3R-SLAM checkout e6f4e3d474fad0e11f561482012be864ba8c3f17
PY="$PWD/.venv-mast3r/bin/python"
"$PY" -m pip install 'setuptools<76' wheel cython roma scikit-learn matplotlib tensorboard natsort pyrealsense2 plyfile evo faiss-cpu
"$PY" -m pip install --no-build-isolation 'lietorch @ git+https://github.com/princeton-vl/lietorch.git'
"$PY" -m pip install --no-deps --no-build-isolation -e vendor/MASt3R-SLAM/thirdparty/mast3r
"$PY" -m pip install --no-build-isolation vendor/MASt3R-SLAM/thirdparty/mast3r/dust3r/croco/models/curope
"$PY" -m pip install --no-build-isolation vendor/MASt3R-SLAM/thirdparty/mast3r/asmk
"$PY" scripts/prepare-comparison.py
"$PY" -m pip install --no-deps --no-build-isolation -e vendor/MASt3R-SLAM
"$PY" -m pip install numpy==1.26.4 plyfile==1.1.3 contourpy==1.3.2
"$PY" -m pip freeze > artifacts/mast3r-requirements-lock.txt
