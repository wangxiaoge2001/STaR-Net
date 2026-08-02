"""ONNX-export-only, pure PyTorch implementation of a pretrained Mamba2 layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExportableMamba2(nn.Module):
    """Evaluate Mamba2 with an explicit recurrent scan for fixed-shape export.

    The official fused CUDA/Triton path is much faster in PyTorch but cannot be
    traced by the legacy ONNX exporter. This wrapper reuses every pretrained
    parameter and expresses the same recurrence using standard PyTorch ops.
    It is intended only for exporting an in-memory model copy; training and
    ordinary PyTorch inference should continue to use the official Mamba2.
    """

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source

    def forward(self, u):
        m = self.source
        batch, seqlen, _ = u.shape
        projected = m.in_proj(u)
        A = -torch.exp(m.A_log.float())
        z, xBC, dt = torch.split(
            projected,
            [m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads],
            dim=-1,
        )

        xBC = F.conv1d(
            xBC.transpose(1, 2),
            m.conv1d.weight,
            m.conv1d.bias,
            padding=m.d_conv - 1,
            groups=xBC.shape[-1],
        )[:, :, :seqlen].transpose(1, 2)
        xBC = F.silu(xBC)
        x, B, C = torch.split(
            xBC,
            [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state],
            dim=-1,
        )
        x = x.view(batch, seqlen, m.nheads, m.headdim)
        B = B.view(batch, seqlen, m.ngroups, m.d_state).repeat_interleave(
            m.nheads // m.ngroups, dim=2
        )
        C = C.view(batch, seqlen, m.ngroups, m.d_state).repeat_interleave(
            m.nheads // m.ngroups, dim=2
        )
        dt = F.softplus(dt + m.dt_bias)

        state = torch.zeros(
            (batch, m.nheads, m.headdim, m.d_state),
            dtype=x.dtype,
            device=x.device,
        )
        outputs = []
        for index in range(seqlen):
            decay = torch.exp(dt[:, index, :, None, None] * A[None, :, None, None])
            update = (
                x[:, index, :, :, None]
                * B[:, index, :, None, :]
                * dt[:, index, :, None, None]
            )
            state = state * decay + update
            y = (state * C[:, index, :, None, :]).sum(-1)
            y = y + x[:, index] * m.D[None, :, None]
            outputs.append(y)

        y = torch.stack(outputs, dim=1).reshape(batch, seqlen, m.d_ssm)
        z = z.reshape(batch, seqlen, m.d_ssm)
        if m.norm_before_gate:
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + m.norm.eps)
            y = y * m.norm.weight * F.silu(z)
        else:
            y = y * F.silu(z)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + m.norm.eps)
            y = y * m.norm.weight
        return m.out_proj(y)


def replace_mamba2_for_export(model: nn.Module) -> int:
    """Replace every official Mamba2 child with :class:`ExportableMamba2`."""
    from mamba_ssm import Mamba2

    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, Mamba2):
            setattr(model, name, ExportableMamba2(child))
            replaced += 1
        else:
            replaced += replace_mamba2_for_export(child)
    return replaced
