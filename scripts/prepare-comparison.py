# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Portions adapted from MASt3R-SLAM; copyright 2025-present its authors.
# Adaptations copyright 2026 abot-realtime-map contributors.
# Changes: headless imports, runtime reporting, and SM89 compilation settings.
# See THIRD_PARTY_NOTICES.md and licenses/MASt3R-SLAM/LICENSE.md.
"""Headless import/SM89 build adaptations only; no tracking algorithm changes."""
from pathlib import Path
import json
import re
root=Path('vendor/MASt3R-SLAM')
main=root/'main.py'
text=main.read_text()
old='from mast3r_slam.visualization import WindowMsg, run_visualization'
new='''# Headless adaptation: the original WindowMsg data, no OpenGL imports.
from dataclasses import dataclass
@dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5
def run_visualization(*args, **kwargs):
    from mast3r_slam.visualization import run_visualization as run
    return run(*args, **kwargs)
'''
if old in text:main.write_text(text.replace(old,new))
text=main.read_text()
marker='    if save_frames:\n        savedir = '
instrument='''    import json
    torch.cuda.synchronize()
    (save_dir / "runtime.json").write_text(json.dumps({
        "frames": len(dataset), "seconds_including_backend_startup": time.time() - fps_timer,
        "keyframes": len(keyframes), "single_thread": config["single_thread"],
        "peak_frontend_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        "note": "Headless unpaced throughput; includes final result export, excludes model loading. Not browser end-to-end latency."
    }, indent=2))
'''
if 'runtime.json' not in text:main.write_text(text.replace(marker,instrument+marker))
setup=root/'setup.py'; text=setup.read_text()
text=re.sub(r'"-gencode=arch=compute_\d+,code=sm_\d+",\n', '', text)
text=text.replace('extra_compile_args["nvcc"] = [','extra_compile_args["nvcc"] = ["-gencode=arch=compute_89,code=sm_89",')
setup.write_text(text)
(root/'config/realtime-map.yaml').write_text('inherit: config/base.yaml\nuse_calib: False\nsingle_thread: True\ndataset:\n  subsample: 1\n')
