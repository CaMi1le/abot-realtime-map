# Predicted OCC conditioning for ABot

The direction is **camera OCC predictions → ABot internal patch tokens → geometry/pose/confidence**.
The default RGB-only path is unchanged. OD, PSD and externally exported BEV
features are not required. This is an experimental, trainable integration; it
does not ship trained Waymo OCC/adapter weights or establish a quality/latency gain.

## What runs

`CameraOccupancyNet` is a new compact camera-to-voxel baseline using the parking
pipeline's architectural idea: shared image features, calibrated projection to
a 4-layer feature volume, and a BEV decoder producing 10 occupancy layers.
It uses Waymo's pinhole/Brown-Conrady calibration, not the parking fisheye
projection. Its parameters are **not** a conversion of MagModel's checkpoint.
The two-class head predicts free/occupied probabilities; outside camera support
remains unknown. One-camera and front-three checkpoints are trained separately.
`prediction_from_logits` also normalizes an existing external OCC head, including
parking free/occupied/wall logits, into this same voxel interface. Its calibration
and domain adaptation remain the responsibility of that predictor.

`VoxelProjector` uses the FRONT calibration to raycast predictions at image patch
centres. It accounts for ABot's width resize, centre crop, padding, and distortion.
The four hints are first-hit log camera-z depth in metres, hit probability,
visible free probability and known fraction. Ray spacing is at most half a voxel;
storage is bounded by the grid and one current camera geometry.

`OccTokenAdapter` adds a zero-initialized residual to `x_norm_patchtokens` immediately
after the existing image encoder. The existing streaming decoder then consumes
these tokens and writes its usual causal KV state. Token count/register tokens/
position indices remain unchanged. The scoped hook runs on both SDPA and paged
inference paths, without editing ignored upstream code or running ABot twice.
`zero` and `shuffle` still execute OCC prediction/projection, so their runtime
comparisons retain those costs. Shuffle permutes patches within the same frame;
it never takes information from a future frame.

The model-unit scale of ABot remains uncalibrated. Metric OCC hints do not by
themselves prove that the frozen camera/depth heads acquire metric scale.

## Data and training

Install the optional `occ` dependencies in an isolated environment alongside
the pinned ABot installation documented in the main README. The CPU verification
environment additionally uses the official PyTorch 2.5.1/torchvision 0.20.1 CPU wheels.

Use the existing `reconstruction` export containing `source-manifest.json`,
`images/`, fixed `extrinsics/<camera_id>.txt`, and offline `lidar/` Nx14 files.
The loader consumes only images, fixed calibration, timestamps and official
context identities. It never reads world poses, LiDAR or object labels online.
Raw distorted RGB is retained and projection uses its distortion model.
Exposure/rolling-shutter metadata is preserved by the exporter; this initial
integration uses fixed extrinsics, not rolling-shutter motion compensation.

Each official context is an independent sequence. `--waymo-context` is required
when a manifest contains multiple contexts. Train/validation contexts are disjoint;
related nearby contexts can still share scenery, so context separation alone is
not evidence of route-level generalization. The older local 5-frame/15-image sample
has one frame per context, **not five consecutive video frames**. A separate native
preview now contains 20 consecutive timestamps (about 1.9 seconds at 10 Hz) from
context `9653249092275997647_980_000_1000_000`, with all three original 1920×1280
JPEGs per timestamp. The prepared training root below adds measured LiDAR to those
20 frames; its validation root contains one native first frame from a different,
adjacent context. These small samples support wiring checks; they cannot establish
route-level generalization, tracking quality or sustained processing capacity.

The original two-stage training remains available. Train the source first, with
`CONTEXT_VAL` replaced by an actual held-out context:

```bash
python -m realtime_map.occ_training --stage occupancy \
  --input /path/to/waymo/scene --validation-context CONTEXT_VAL \
  --cameras 1 --steps 1000 --device cuda --output models/waymo-occ-front.pt

python -m realtime_map.occ_training --stage fusion \
  --input /path/to/waymo/scene --validation-context CONTEXT_VAL \
  --cameras 1 --steps 1000 --device cuda \
  --occ-checkpoint models/waymo-occ-front.pt \
  --output models/waymo-occ-adapter-front.pt
```

The step counts are example smoke/research settings, not convergence criteria.
For front-three, train both stages with `--cameras 3` and separate output paths.
The initial source ROI is `[-10,-10,-0.5]` to `[10,10,1.5]` metres, 0.2m voxels,
so this first experiment tests near-field conditioning rather than whole-road
reconstruction. Targets contain measured LiDAR return endpoints and free cells
along measured rays; unobserved cells are ignored, and occupied endpoints win.
Boxes are not filled as dense OCC truth.

Fusion training freezes OCC and all ABot parameters and optimizes only the
adapter through the causal SDPA path. Its initial loss is scale-invariant sparse
camera-z depth; it is not an object tracking loss. KV, reference-hidden and camera
state are detached between steps and reset at context, timestamp-rewind and epoch
boundaries. Validation starts with fresh state and is independent of the training
optimizer. Checkpoints retain the source camera set, training and
validation contexts and actual completed steps. Adapters are bound to the SHA256
of the OCC checkpoint they were trained against. Runtime rejects untrained or
incompatible checkpoints instead of using random weights as a useful prior.

## Joint training with the front three cameras

`realtime_map.occ_joint_training` jointly optimizes the image OCC predictor and
zero-initialized token adapter. OCC uses FRONT, FRONT_LEFT and FRONT_RIGHT by
default; ABot still consumes FRONT once per timestamp. ABot's parameters stay
frozen, while its forward pass remains differentiable so the reconstruction loss
can reach the adapter and OCC predictor. This is the compact pinhole OCC baseline
described above; the original parking weights have not been converted.

The default objective is observed-voxel cross entropy plus scale-invariant sparse
camera-z depth loss, with weights controlled by `--occ-weight` and `--depth-weight`.
Unknown LiDAR space is ignored. LiDAR and world poses remain offline metadata and
supervision; online OCC predictions consume only the three images and fixed
calibration. Scale-invariant depth supervision does not establish metric ABot scale.

Joint training uses `SoftVoxelProjector`: calibrated rays traverse the intersected
voxels and accumulate probabilistic first-hit depth, hit probability, free-space
evidence and known support. This permits gradients through OCC probabilities.
The exported `realtime-map-occ-adapter-v2` checkpoint records the projection
configuration, and runtime restores the same soft projection. It also binds the
adapter to both the exported OCC and frozen ABot checkpoint hashes. Original
adapter-v1 checkpoints retain their original hard first-hit projection.

The zero-initialized adapter initially preserves the ABot baseline, so its first
optimizer update has zero reconstruction-to-OCC gradient. OCC still learns from
its own supervised objective. After the adapter has updated, reconstruction can
also influence OCC. `--gradient-audit` records this gradient and both the
reconstruction-only and joint-loss image gradients for each camera; it adds backward work and should be disabled for
performance measurements.

Prepare the available local TFRecords without fetching data or modifying their
contents. Run these commands from `realtime-map/`; output roots must be new:

```bash
python scripts/prepare-waymo-supervision.py \
  --source data/waymo-detail/front-context-9653249092275997647/source-prefix.tfrecord \
  --output data/waymo-joint/train-context-9653249092275997647 --max-frames 20

python scripts/prepare-waymo-supervision.py \
  --source ../reconstruction/artifacts/waymo-plan-20261014/candidate-source/02_7761658966964621355_1000_000_1020_000_first.tfrecord \
  --output data/waymo-joint/validation-context-7761658966964621355 --max-frames 1
```

These two roots have already been prepared in this workspace. The exporter reuses
the workspace's TensorFlow-free decoder, verifies TFRecord CRCs, retains original
JPEG bytes, calibration, timestamps and source hashes, and writes measured Nx14
LiDAR rays. It preserves TOP per-pixel deskew and the first return from all five
LiDAR sensors. Ground labels are explicitly unknown (`-1`) and are not used by
the training losses. Incomplete, mixed-context or nonmonotonic inputs are rejected
before the output root is published.

Run a small CPU training check using those prepared roots:

```bash
python -m realtime_map.occ_joint_training \
  --input data/waymo-joint/train-context-9653249092275997647 \
          data/waymo-joint/validation-context-7761658966964621355 \
  --validation-context 7761658966964621355_1000_000_1020_000 \
  --cameras 3 --steps 2 --clip-frames 2 --gradient-audit --device cpu \
  --output artifacts/waymo-joint/joint3-smoke
```

`--steps` counts optimizer updates over clips, not individual frames. Each clip
uses increasing timestamps from one official context, with causal ABot state.
History is detached after every frame, so this is not full backpropagation through
time. State resets between optimizer clips, context boundaries and evaluation
ablations; stale states are not reused after a parameter update.

`state.pt` stores OCC and adapter parameters, optimizer state, RNG state, completed
updates, deterministic clip position and dataset/checkpoint identities. Resume is
at a completed clip boundary; historical KV tensors do not need to be serialized.
The same inputs, split, frozen ABot weights and training configuration are required.
For example, continue the two-update check to **three total updates** in a new
output directory:

```bash
python -m realtime_map.occ_joint_training \
  --input data/waymo-joint/train-context-9653249092275997647 \
          data/waymo-joint/validation-context-7761658966964621355 \
  --validation-context 7761658966964621355_1000_000_1020_000 \
  --cameras 3 --steps 3 --clip-frames 2 --gradient-audit --device cpu \
  --resume artifacts/waymo-joint/joint3-smoke/state.pt \
  --output artifacts/waymo-joint/joint3-resumed
```

The completed run exports `occ.pt`, `adapter.pt`, `training.jsonl` and `report.json`,
alongside the resumable state. Validation reports predicted, zero and shuffled OCC
ablations. Existing inference flags load the exported pair:

```bash
realtime-map --dataset-format waymo \
  --input data/waymo-joint/validation-context-7761658966964621355 \
  --waymo-context 7761658966964621355_1000_000_1020_000 \
  --device cpu --attention sdpa --max-frames 1 \
  --occ-checkpoint artifacts/waymo-joint/joint3-smoke/occ.pt \
  --occ-adapter artifacts/waymo-joint/joint3-smoke/adapter.pt \
  --occ-mode predicted --output artifacts/waymo-joint/joint3-inference
```

These commands demonstrate the training and checkpoint workflow; their short
step counts are not convergence settings. Twenty training frames and one adjacent
held-out-context frame do not support claims of reconstruction improvement,
tracking gains or real-time performance. Broader independent contexts and target
GPU measurements remain necessary.

## Matched causal runs

Run one complete context for each profile, using identical settings. Add the two
checkpoint flags for the conditioned run; omit them for the baseline:

```bash
realtime-map --dataset-format waymo --input /path/to/waymo/scene \
  --waymo-context CONTEXT_TEST --output artifacts/front-occ \
  --attention paged --realtime --viewer --measure-browser \
  --occ-checkpoint models/waymo-occ-front.pt \
  --occ-adapter models/waymo-occ-adapter-front.pt --occ-mode predicted

python -m realtime_map.occ_experiment artifacts/front-base artifacts/front-occ \
  --output artifacts/front-occ/comparison.json
python -m realtime_map.waymo_evaluation artifacts/front-occ /path/to/waymo/scene \
  --output artifacts/front-occ/waymo-evaluation.json
```

Repeat with `--occ-mode zero`, `--occ-mode shuffle`, and separately trained
front-three checkpoints. `model_ms` includes the adapter. `occ_predict_ms`
includes image loading/preprocessing and online OCC inference; `occ_project_ms`
includes ray projection and ablation. Total processing and arrival-to-render
latency include every stage. There is no precomputed-prior realtime shortcut.
The comparison tool rejects unmatched inputs/hardware/runtime settings, and
short smoke runs cannot pass sustained-cadence acceptance. Its backlog/service
gate is diagnostic; inspect browser P95 and quality together before adoption.

Waymo evaluation fits one global trajectory Sim(3) offline and reports ATE and
observed sparse depth errors. Ground truth is not used to set inference scale.
Tracking remains a separate experiment: ABot emits no IDs. Keep proposal and
association methods fixed across all profiles; `association_scores` provides an
association-only switch diagnostic for independently matched objects, and does
not claim official tracking metrics. A complete object detector/tracker is not
provided by token conditioning.

## Verification and remaining evidence

Verification in this workspace (CPU, PyTorch 2.5.1): **56 tests passed**. The
released ABot checkpoint and real Waymo FRONT image gave exact point/pose parity
with a zero prior. One real-data optimization step completed for the independent
OCC source and then for the fusion adapter with ABot/OCC frozen. Reloaded predicted
OCC changed the real ABot point outputs; zeroing the hints restored exact baseline
outputs. This establishes wiring and trainability, not an improvement in quality.
The conditioned command-line run completed and saved dense observations, map,
poses and stage timings. Local evidence is in `artifacts/occ-verification/`.

On 2026-10-23, the three-camera joint path completed two real optimizer updates
of two consecutive frames each, then resumed `state.pt` for a third two-frame
update. Thus six of the twenty available training frames were used in this wiring
check. OCC has 94,804 trainable parameters and the adapter 66,880; all ABot
parameters remained frozen. The second update produced nonzero reconstruction-only
gradients to OCC logits. The resumed update also measured nonzero reconstruction-only
image gradients for all three real calibrated camera views. Optimizer counters
advanced from 2 to 3 and both OCC and adapter weights changed after resume.

The held-out context ran predicted/zero/shuffled validation. Exported weights also
completed the existing command-line inference path, with online OCC generation
included in timings. A released-weight comparison confirmed exact point/pose parity
between zero hints and unconditioned ABot; predicted hints changed point outputs.
Training/inference soft-projection parity, frozen-weight bindings and context-safe
supervision lookup are covered by tests. Evidence is in
`artifacts/waymo-joint/joint3-smoke/`, `joint3-resumed/` (including
`resume-verification.json` and `real-inference-parity.json`), and
`artifacts/waymo-joint/joint3-inference/`. These CPU checks establish the gradient,
checkpoint and inference paths, not improved geometry/tracking or GPU realtime
performance. The single adjacent-context validation frame is insufficient for
generalization claims.

Reproduce the real-weight integration check with trained or smoke checkpoints:

```bash
python scripts/verify-occ.py --input /path/to/waymo/scene --context CONTEXT_TEST \
  --occ-checkpoint models/waymo-occ-front.pt \
  --occ-adapter models/waymo-occ-adapter-front.pt --device cpu \
  --output artifacts/occ-verification/real-conditioned-inference.json
```

Tests cover voxel first hits, unknown/free distinctions, patch geometry, camera
axes, context separation, one/three-view predictor gradients, token conditioning,
causal state influence, no-prior/zero initialization parity and adapter gradients.
CPU synthetic tests do not establish CUDA paged-attention correctness or realtime
performance; verify both with the released ABot weights on the target GPU.
The local real samples can validate calibration and checkpoint parity. Continuous
held-out Waymo clips and trained checkpoints are needed for the planned quality,
tracking and sustained realtime comparisons.
