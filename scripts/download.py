# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 abot-realtime-map contributors.
"""Fetch public, version-pinned model and original TUM benchmark archives."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ["freiburg1_desk", "freiburg1_room", "freiburg3_walking_xyz"]


def dataset(name):
    folder = ROOT / "data" / f"rgbd_dataset_{name}"
    if (folder / "rgb.txt").is_file(): return
    archive = ROOT / "data" / f"{name}.tgz"
    url = f"https://cvg.cit.tum.de/rgbd/dataset/{name.split('_')[0]}/rgbd_dataset_{name}.tgz"
    if not archive.is_file():
        tmp = archive.with_suffix(".partial")
        print("Downloading", url, flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(ROOT / "data", filter="data")
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    (folder / "source.json").write_text(json.dumps({"url": url, "sha256": digest}, indent=2))
    print("Ready", name, flush=True)


def model():
    path = hf_hub_download("acvlab/ABot-Recon", "abot_recon.safetensors",
                          revision="c69c26ca1853afc9c9212014459e729d7339ecdf",
                          local_dir=ROOT / "models")
    digest = hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()
    assert digest == "ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e"
    print("Checkpoint SHA256 verified", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(model)] + [pool.submit(dataset, name) for name in SEQUENCES]
        for future in futures: future.result()
