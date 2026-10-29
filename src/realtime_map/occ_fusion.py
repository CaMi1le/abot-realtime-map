# SPDX-License-Identifier: Apache-2.0
"""OCC conditions existing patch tokens before ABot's streaming decoder."""

from contextlib import contextmanager

import torch
from torch import nn


class OccTokenAdapter(nn.Module):
    def __init__(self, token_dim=1024, hidden_dim=64):
        super().__init__()
        self.token_dim, self.hidden_dim = token_dim, hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim)
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, tokens, hints):
        if hints.shape != (*tokens.shape[:2], 4) or tokens.shape[-1] != self.token_dim:
            raise ValueError("OCC patches must match ABot's encoder token layout")
        hints = hints.to(device=tokens.device, dtype=self.projection[0].weight.dtype)
        if not torch.isfinite(hints).all():
            raise ValueError("Non-finite OCC patch features")
        update = self.projection(hints) * hints[..., 3:4].clamp(0, 1)
        return tokens + update.to(tokens.dtype)


class EncoderConditioner:
    """Scoped encoder hook works on both SDPA and paged paths, including training.

    Never changes the encoder's class/register tokens, sequence length or KV
    layout. A missing prior bypasses the adapter. Close removes the hook.
    """

    def __init__(self, encoder, adapter):
        self.adapter = adapter
        self.pending = None
        self.calls = 0
        self.handle = encoder.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        hints = self.pending
        if hints is None:
            return output
        self.pending = None
        self.calls += 1
        if isinstance(output, dict):
            if "x_norm_patchtokens" not in output:
                raise ValueError("Unsupported ABot encoder output")
            return {
                **output,
                "x_norm_patchtokens": self.adapter(output["x_norm_patchtokens"], hints),
            }
        return self.adapter(output, hints)

    @contextmanager
    def condition(self, hints):
        if self.pending is not None:
            raise RuntimeError("Nested OCC conditioning is not supported")
        self.pending = hints
        before = self.calls
        try:
            yield
            if hints is not None and self.calls != before + 1:
                raise RuntimeError(
                    "OCC must condition exactly one encoder forward per frame"
                )
        finally:
            self.pending = None

    def close(self):
        self.pending = None
        self.handle.remove()
