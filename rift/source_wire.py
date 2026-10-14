"""Packed predictive residuals; transforms depend only on public message identifiers."""
from __future__ import annotations
import hashlib
import math
import struct
from typing import Mapping
import numpy as np
import torch
from . import rotation as gaussian

KIND_GAUSSIAN_4LEVEL = 5
KIND_GAUSSIAN_ROTATED_4LEVEL = 6
VARIANT_TO_KIND = {"gaussian4": 5, "gaussian4_rotated": 6}

def _variant(variant: str) -> tuple[int, bool]:
    if variant not in VARIANT_TO_KIND:
        raise ValueError(f"unknown Stage B codec variant: {variant}")
    return VARIANT_TO_KIND[variant], variant == "gaussian4_rotated"


def _public_seed(variant: str, round_id: int, sender_id: int, entry_index: int) -> int:
    """Derive rotation signs only from public frame identity, never training RNG."""
    token = (f"round30-stage-b-public-transform:v1:{variant}:"
             f"{int(round_id)}:{int(sender_id)}:{int(entry_index)}").encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def pack_symbols(codes: np.ndarray) -> bytes:
    values = np.asarray(codes)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Gaussian symbols must be a one-dimensional integer array")
    if values.size == 0 or int(values.min()) < 0 or int(values.max()) > 3:
        raise ValueError("Gaussian symbols must be nonempty values in [0, 3]")
    packed = bytearray((values.size + 3) // 4)
    for index, value in enumerate(values):
        packed[index // 4] |= int(value) << ((index % 4) * 2)
    return bytes(packed)


def unpack_symbols(payload: bytes, count: int) -> np.ndarray:
    if count <= 0 or len(payload) != (count + 3) // 4:
        raise ValueError("Gaussian symbol payload length does not match coefficient count")
    values = np.empty(count, dtype=np.uint8)
    for index in range(count):
        values[index] = (payload[index // 4] >> ((index % 4) * 2)) & 0x3
    if count % 4 and payload and payload[-1] >> ((count % 4) * 2):
        raise ValueError("Gaussian symbol payload has nonzero padding")
    return values


def _serialize(protocol, variant: str, values: Mapping[str, torch.Tensor], specs: Mapping,
               round_id: int, sender_id: int) -> tuple[bytes, dict[str, torch.Tensor]]:
    kind, rotate = _variant(variant)
    if tuple(values) != tuple(specs):
        raise ValueError("Gaussian frame tensor order differs from public schema")
    entries = []
    decoded: dict[str, torch.Tensor] = {}
    for entry_index, (name, spec) in enumerate(specs.items()):
        raw = values[name]
        if not torch.is_tensor(raw):
            raise TypeError("Gaussian frame values must be tensors")
        allowed_shapes = {tuple(spec.tensor_shape), (int(spec.coefficient_count),)}
        if tuple(raw.shape) not in allowed_shapes:
            raise ValueError(f"Gaussian coefficient shape mismatch for {name}")
        value = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
        if value.numel() != int(spec.coefficient_count) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Gaussian coefficient count or finiteness invalid for {name}")
        seed = _public_seed(variant, round_id, sender_id, entry_index)
        codes, scale, _metadata = gaussian.encode(
            value.numpy(), rotate=rotate, public_seed=seed,
            block_size=gaussian.DEFAULT_BLOCK_SIZE, diagnostics=False,
        )
        packed = pack_symbols(codes)
        scale_bytes = struct.pack("<f", float(np.float32(scale)))
        entries.append(protocol._entry_bytes(entry_index, spec, packed + scale_bytes))
        estimate = gaussian.decode(
            codes, scale, rotate=rotate, public_seed=seed,
            block_size=gaussian.DEFAULT_BLOCK_SIZE,
        )
        decoded[name] = torch.from_numpy(np.asarray(estimate, dtype=np.float32).copy())
    frame = protocol._frame_header(kind, round_id, sender_id, protocol.SERVER_ID, len(entries))
    return frame + b"".join(entries), decoded


def _decode(protocol, variant: str, payload: bytes, specs: Mapping,
            round_id: int, sender_id: int) -> dict[str, torch.Tensor]:
    expected_kind, rotate = _variant(variant)
    parts = protocol._parse_frame(
        payload, kind=expected_kind, round_id=round_id, sender_id=sender_id,
        receiver_id=protocol.SERVER_ID, specs=specs,
    )
    decoded: dict[str, torch.Tensor] = {}
    for entry_index, ((name, spec), data) in enumerate(zip(specs.items(), parts)):
        count = int(spec.coefficient_count)
        packed_count = (count + 3) // 4
        if len(data) != packed_count + 4:
            raise ValueError(f"Gaussian scale or code length invalid for {name}")
        codes = unpack_symbols(data[:packed_count], count)
        scale = struct.unpack("<f", data[packed_count:])[0]
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"Gaussian scale invalid for {name}")
        if scale == 0.0 and np.any(codes):
            raise ValueError(f"zero-scale Gaussian tensor has nonzero symbols for {name}")
        seed = _public_seed(variant, round_id, sender_id, entry_index)
        value = gaussian.decode(
            codes, scale, rotate=rotate, public_seed=seed,
            block_size=gaussian.DEFAULT_BLOCK_SIZE,
        )
        decoded[name] = torch.from_numpy(np.asarray(value, dtype=np.float32).copy())
    return decoded

