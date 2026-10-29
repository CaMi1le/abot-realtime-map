# SPDX-License-Identifier: Apache-2.0
"""Separate fusion training must preserve the deployed streaming state contract."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from realtime_map.waymo import WaymoFrame


def assert_detached(value):
    if torch.is_tensor(value):
        assert not value.requires_grad and value.grad_fn is None
    elif isinstance(value, dict):
        for child in value.values():
            assert_detached(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_detached(child)


class FusionEncoder(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.embed_dim)

    def forward(self, image):
        tokens = self.projection(image.mean(dim=(-2, -1)))[:, None]
        return {"x_norm_patchtokens": tokens}


class FusionNetwork(nn.Module):
    def __init__(self, reference_output):
        super().__init__()
        self.encoder = FusionEncoder()
        self.decoder = nn.Linear(8, 14 * 14)
        self.reference_output = reference_output
        self.arguments = []
        self.references = []

    def forward(self, images, **kwargs):
        self.arguments.append(kwargs)
        assert kwargs["causal_global_attn"]
        assert kwargs["long_sequence_parallel"] and kwargs["streaming_inference"]
        tokens = self.encoder(images[:, 0])["x_norm_patchtokens"]
        reference = kwargs["ref_hidden"]
        if reference is not None:
            tokens = tokens + 0.1 * reference
        depth = self.decoder(tokens).exp().reshape(1, 1, 14, 14)
        output = {
            "local_points": torch.stack((depth * 0, depth * 0, depth), -1),
            "past_key_values": [(tokens[:, None], tokens[:, None] * 2)],
            "camera_state": {"pose": tokens.sum(-1), "nested": (tokens, [tokens * 2])},
        }
        # A head can emit a reference only on the first frame, or repeat it.
        if reference is None or self.reference_output == "repeat":
            output["ref_hidden"] = tokens if reference is None else reference
        elif self.reference_output == "none":
            output["ref_hidden"] = None
        self.references.append(output.get("ref_hidden"))
        return output

    def _postprocess_stream_carry(self, carry, spatial_hw):
        assert spatial_hw == (14, 14)
        assert_detached(carry)
        return carry


@pytest.mark.parametrize("reference_output", ["repeat", "omitted", "none"])
def test_separate_fusion_carries_detached_state_and_resets_train_and_validation(
    monkeypatch, tmp_path, reference_output
):
    import realtime_map.backend as backend_module
    import realtime_map.occ_training as training

    preprocessing = pytest.importorskip("abot_recon.preprocessing")
    transform = SimpleNamespace(target_height=14, target_width=14)
    monkeypatch.setattr(
        preprocessing,
        "preprocess_image",
        lambda image: (
            torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255,
            transform,
        ),
    )
    monkeypatch.setattr(
        training,
        "depth_samples",
        lambda *args: (
            torch.arange(14 * 14),
            torch.linspace(1, 3, 14 * 14),
        ),
    )

    class Backend:
        def __init__(self):
            self.device = "cpu"
            self.wrapper = SimpleNamespace(compute_dtype=torch.float32)
            self.network = FusionNetwork(reference_output)
            self.resets = 0
            self.closed = False

        def reset(self):
            self.resets += 1

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr(backend_module, "ABotStream", lambda *args, **kwargs: backend)
    source = SimpleNamespace(
        cameras=("FRONT",),
        metadata={
            "training": {"contexts": ["occ-train"]},
        },
    )
    monkeypatch.setattr(training, "PredictedOccSource", lambda *args: source)
    monkeypatch.setattr(
        training,
        "OccRuntime",
        lambda source, adapter: SimpleNamespace(
            prepare=lambda frame, transform: (torch.ones(1, 1, 4), {}),
        ),
    )

    image = tmp_path / "frame.png"
    Image.new("RGB", (14, 14), (60, 90, 120)).save(image)
    first = WaymoFrame(0, 1.0, image, "train-a", 1_000_000, ())

    def sequence(prefix):
        return [
            (
                replace(
                    first,
                    frame_id=index,
                    context=context,
                    timestamp=timestamp,
                    timestamp_us=round(timestamp * 1e6),
                ),
                tmp_path,
                {},
            )
            for index, (context, timestamp) in enumerate(
                [
                    (prefix + "-a", 1.0),
                    (prefix + "-a", 1.1),
                    (prefix + "-a", 1.2),
                    (prefix + "-b", 2.0),
                    (prefix + "-b", 1.9),
                ]
            )
        ]

    occ_checkpoint = tmp_path / "occ.pt"
    occ_checkpoint.write_bytes(b"predictor identity")
    args = SimpleNamespace(
        abot_checkpoint="unused.pt",
        occ_checkpoint=occ_checkpoint,
        device="cpu",
        cameras=1,
        lr=1e-3,
        steps=6,
        output=tmp_path / "adapter.pt",
    )
    result = training.train_fusion(args, sequence("train"), sequence("validation"))
    assert math.isfinite(result["scale_invariant_log_depth_loss"])
    expected_history = [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert len(backend.network.arguments) == len(expected_history)
    for index, (kwargs, expected) in enumerate(
        zip(
            backend.network.arguments,
            expected_history,
            strict=True,
        )
    ):
        for key in ("past_key_values", "ref_hidden", "camera_state"):
            assert (kwargs[key] is not None) == expected, (index, key)
            assert_detached(kwargs[key])
    for start in (0, 6):
        for index in (start + 1, start + 2):
            torch.testing.assert_close(
                backend.network.arguments[index]["ref_hidden"],
                backend.network.references[start],
            )
    # The training reference originally had a graph: detachment is substantive.
    assert backend.network.references[0].grad_fn is not None
    assert all(parameter.grad is None for parameter in backend.network.parameters())
    assert backend.resets == 7 and backend.closed
    assert not backend.network.encoder._forward_hooks
    saved = torch.load(args.output, weights_only=True)
    assert saved["training"]["steps"] == 6
    assert saved["validation"] == result
