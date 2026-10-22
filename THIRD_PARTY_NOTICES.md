# Third-party notices and license boundaries

Reviewed against the pinned source trees on 2026-10-22. The root [Apache-2.0 license](LICENSE) covers this project's original code and documentation unless a file states otherwise. It does not relicense downloaded software, checkpoints, datasets, or their derivatives. The Python package contains the Apache-2.0 application code; an optional baseline adaptation script has a separate license.

## License inventory

| Component | Source / version | Applicable terms | Distribution here |
|---|---|---|---|
| Project application, tests, original tools and docs | This repository | Apache-2.0, except the file below | Source included |
| `scripts/prepare-comparison.py` | Adapted from MASt3R-SLAM at `e6f4e3d474fad0e11f561482012be864ba8c3f17` | **CC BY-NC-SA 4.0** | Marked adaptation script included |
| ABot-Recon code | `6e3b2e1257464a74dfceff22eb33190bd228276e` | Apache-2.0 overall; file-specific exceptions below | Downloaded separately; adapter provenance in NOTICE |
| ABot-Recon weights | `acvlab/ABot-Recon`, revision `c69c26ca1853afc9c9212014459e729d7339ecdf` | **CC BY-NC 4.0**, plus applicable upstream Pi3 terms | Not included |
| Pi3 code / weights | Components identified by pinned ABot-Recon | BSD-3-Clause code; CC BY-NC 4.0 released weights | Obtained through upstream; not bundled |
| cuRoPE2D / CroCo / DUSt3R CUDA sources | ABot-Recon `abot_recon/modeling/pi3/models/curope/` | **CC BY-NC-SA 4.0**; NAVER and upstream notices apply | Built only from separate upstream checkout |
| FlashInfer | `0.2.5` in acceleration setup | Apache-2.0 | External dependency |
| MASt3R-SLAM code | `e6f4e3d474fad0e11f561482012be864ba8c3f17` | **CC BY-NC-SA 4.0** | Separate optional checkout |
| MASt3R / DUSt3R / CroCo checkpoints | Optional baseline's upstream assets | Their licenses and checkpoint notices, including training-data terms | Not included |
| DINOv2 / SALAD checkpoints | Upstream loop-asset downloader | Respective publisher/checkpoint terms | Not included |
| KITTI and TUM RGB-D | Official benchmark sources | Independent dataset terms and attribution requirements | No data, ground-truth files, maps, or recordings included |

Installed Python/npm packages retain their own licenses. The table focuses on the reconstruction components and exceptions; it is not a replacement for dependency license files or a complete binary-distribution bill of materials.

## ABot-Recon attribution and modifications

- Upstream: [AMAP CV Lab / ABot-Recon](https://github.com/amap-cvlab/ABot-Recon/tree/6e3b2e1257464a74dfceff22eb33190bd228276e).
- Unmodified copies: [LICENSE](licenses/ABot-Recon/LICENSE), [NOTICE](licenses/ABot-Recon/NOTICE), [MODEL_LICENSE.md](licenses/ABot-Recon/MODEL_LICENSE.md), and [THIRD_PARTY_NOTICES.md](licenses/ABot-Recon/THIRD_PARTY_NOTICES.md).
- Adaptation: [backend.py](src/realtime_map/backend.py) follows upstream per-frame inference state transitions and returns CPU observations without accumulating dense GPU outputs. Our [NOTICE](NOTICE) identifies this change. The upstream checkout itself is not modified by this adapter.
- Pi3 attribution: [Pi3](https://github.com/yyfz/Pi3), copyright its authors. ABot-Recon states its core encoder/reconstruction code derives from Pi3 and its weights were initialized from Pi3 weights.

The explicit upstream model license governs the ABot-Recon checkpoint; a broader repository or Hub metadata label must not be read as overriding it. The model notice requires preservation of the license, ABot-Recon and Pi3 attribution, and any additional restrictions accompanying Pi3 weights. It states that commercial use requires separate written authorization from all relevant rights holders.

## Optional non-commercial/share-alike components

**cuRoPE:** the source files identify NAVER Corporation and CC BY-NC-SA 4.0. Enabling the optimized path does not make these files Apache-2.0. Preserve their notices and comply with their terms when using or redistributing them and adaptations. See the [pinned upstream notice](licenses/ABot-Recon/THIRD_PARTY_NOTICES.md) and [CC BY-NC-SA 4.0 legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).

**MASt3R-SLAM:** [upstream code](https://github.com/rmurai0610/MASt3R-SLAM/tree/e6f4e3d474fad0e11f561482012be864ba8c3f17) is CC BY-NC-SA 4.0. Its [license notice](licenses/MASt3R-SLAM/LICENSE.md) is retained. The included [adaptation script](scripts/prepare-comparison.py) is explicitly under that same license. It adapts the upstream `WindowMsg` structure, delays visualization imports, adds runtime reporting, and targets SM 8.9; it does not change tracking logic. Generated modifications to upstream files retain upstream terms. This exception is not covered by the root Apache grant.

MASt3R model rights are separate from MASt3R-SLAM code rights. Review the upstream [MASt3R license](https://github.com/naver/mast3r/blob/main/LICENSE), [NOTICE](https://github.com/naver/mast3r/blob/main/NOTICE), and [CHECKPOINTS_NOTICE](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE), including referenced dataset conditions. The download helper does not grant extra rights.

## Other assets and benchmarks

- [DINOv2](https://github.com/facebookresearch/dinov2) and [SALAD](https://github.com/serizba/salad): optional loop-closure assets are fetched from the original publishers. Review each checkpoint's terms independently.
- [KITTI](https://www.cvlibs.net/datasets/kitti/) and [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset): obtain data from their publishers and follow their license and citation requirements. Dataset terms are not granted by this repository.
- [Viser](https://github.com/nerfstudio-project/viser) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer): external visualization and acceleration dependencies; their own notices remain applicable.

This Git repository publishes aggregate benchmark measurements rather than raw benchmark assets or model checkpoints. Locally exported point clouds and recordings are not automatically relicensed under Apache-2.0; check applicable input, model, and component terms before sharing them.

## Practical scope

The published code can be used under its stated software licenses. The measured pipeline relies on non-commercial weights, and the accelerated/baseline paths include additional non-commercial components. Publishing this repository does not establish an unrestricted commercial pipeline or grant rights controlled by upstream authors.
