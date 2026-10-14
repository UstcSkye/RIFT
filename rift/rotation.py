"""Fixed Gaussian Q2 codebook with public signed block-Hadamard rotation."""
from __future__ import annotations
import math
import hashlib
from functools import lru_cache
from typing import Any
import numpy as np

CODEBOOK_THRESHOLD = 0.9816
CODEBOOK_LEVELS = np.array([-1.51042, -0.45278, 0.45278, 1.51042], dtype=np.float32)
DEFAULT_PUBLIC_SEED = 3020261008
DEFAULT_BLOCK_SIZE = 256

def block_plan(count: int, block_size: int = DEFAULT_BLOCK_SIZE) -> list[tuple[int, int]]:
    if count <= 0 or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("count and block size must be positive; block size must be a power of two")
    blocks: list[tuple[int, int]] = []
    start = 0
    while count - start >= block_size:
        blocks.append((start, block_size))
        start += block_size
    tail = count - start
    while tail:
        size = 1 << (tail.bit_length() - 1)
        blocks.append((start, size))
        start += size
        tail -= size
    if start != count or any(size & (size - 1) or size > block_size for _, size in blocks):
        raise AssertionError("block decomposition is invalid")
    return blocks


def fwht_orthonormal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    size = x.size
    if size == 0 or size & (size - 1):
        raise ValueError("Hadamard transform length must be a positive power of two")
    span = 1
    while span < size:
        groups = x.reshape(-1, 2 * span)
        left = groups[:, :span].copy()
        right = groups[:, span:].copy()
        groups[:, :span] = left + right
        groups[:, span:] = left - right
        span *= 2
    x /= math.sqrt(size)
    return x


@lru_cache(maxsize=8192)
def public_signs(seed: int, block_index: int, block_length: int) -> np.ndarray:
    token = f"round30-hadamard-signs:{seed}:{block_index}:{block_length}".encode("ascii")
    derived_seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(derived_seed))
    return (2 * rng.integers(0, 2, size=block_length, dtype=np.int8) - 1).astype(np.float64)


def rotate_blocks(
    values: np.ndarray,
    seed: int = DEFAULT_PUBLIC_SEED,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.empty_like(x)
    plan = block_plan(x.size, block_size)
    details: list[dict[str, int]] = []
    for block_index, (start, size) in enumerate(plan):
        signs = public_signs(seed, block_index, size)
        output[start:start + size] = fwht_orthonormal(signs * x[start:start + size])
        details.append({"start": start, "length": size})
    return output, details


def inverse_rotate_blocks(
    values: np.ndarray,
    seed: int = DEFAULT_PUBLIC_SEED,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.empty_like(y)
    for block_index, (start, size) in enumerate(block_plan(y.size, block_size)):
        signs = public_signs(seed, block_index, size)
        output[start:start + size] = signs * fwht_orthonormal(y[start:start + size])
    return output


def quantize_normalized(values: np.ndarray) -> np.ndarray:
    """Map normalized coordinates to two-bit Gaussian Lloyd-Max symbols."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("quantizer input must be nonempty and finite")
    # Exact threshold ties choose the nearer-to-zero centroid; exact zero maps
    # to the positive inner centroid as the deterministic signed-zero rule.
    return np.where(
        x < -CODEBOOK_THRESHOLD,
        0,
        np.where(x < 0.0, 1, np.where(x <= CODEBOOK_THRESHOLD, 2, 3)),
    ).astype(np.uint8)


def decode(
    codes: np.ndarray,
    scale: float,
    *,
    rotate: bool,
    public_seed: int = DEFAULT_PUBLIC_SEED,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Decode symbols and inverse-rotate using only wire values and public state."""
    symbols = np.asarray(codes)
    if symbols.ndim != 1 or not np.issubdtype(symbols.dtype, np.integer):
        raise ValueError("codes must be a one-dimensional integer array")
    if symbols.size == 0 or int(symbols.min()) < 0 or int(symbols.max()) > 3:
        raise ValueError("two-bit codes must be nonempty and in [0, 3]")
    scalar = np.float32(scale)
    if not math.isfinite(float(scalar)) or scalar < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    if scalar == 0.0:
        return np.zeros(symbols.size, dtype=np.float32)
    reconstructed = CODEBOOK_LEVELS[symbols.astype(np.uint8)] * scalar
    if rotate:
        reconstructed = inverse_rotate_blocks(
            reconstructed, seed=public_seed, block_size=block_size
        )
    return np.asarray(reconstructed, dtype=np.float32)


def encode(values, *, rotate: bool, public_seed: int = DEFAULT_PUBLIC_SEED,
           block_size: int = DEFAULT_BLOCK_SIZE, diagnostics: bool = False
           ) -> tuple[np.ndarray, np.float32, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("codec input must be nonempty and finite")
    x64 = x.astype(np.float64)
    norm_squared = float(np.sum(x64 * x64, dtype=np.float64))
    if norm_squared == 0.0:
        return np.zeros(x.size, dtype=np.uint8), np.float32(0.0), {
            "allzero": True,
            "rotation": None,
            "normalization_eta": 0.0,
            "norm_squared": 0.0,
            "quantized_inner_product": 0.0,
            "scale_unrounded_float64": 0.0,
            "scale_rule": "S=||x||_2^2 / <u,Q(sqrt(d)*u/||x||_2)>; u=x without rotation or u=R(x) with rotation",
        }

    if rotate:
        working, blocks = rotate_blocks(x64, seed=public_seed, block_size=block_size)
        inverse_error = None
        if diagnostics:
            inverse_error = float(np.max(np.abs(inverse_rotate_blocks(
                working, seed=public_seed, block_size=block_size
            ) - x64)))
        rotation_metadata: dict[str, Any] | None = {
            "public_seed": int(public_seed),
            "block_size_cap": int(block_size),
            "block_decomposition": blocks,
            "padded_coefficients": 0,
            "inverse_error_before_quantization": inverse_error,
            "tail_rule": "greedy descending power-of-two chunks after full fixed-size blocks",
        }
    else:
        working = x64
        rotation_metadata = None

    eta = math.sqrt(x.size / norm_squared)
    codes = quantize_normalized(eta * working)
    quantized = CODEBOOK_LEVELS[codes].astype(np.float64)
    denominator = float(np.sum(working * quantized, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise AssertionError("Gaussian-codebook scale denominator must be finite and positive")
    scale64 = norm_squared / denominator
    scale32 = np.float32(scale64)
    if not math.isfinite(float(scale32)) or scale32 <= 0.0:
        raise AssertionError("Gaussian-codebook scale does not fit a positive finite FP32")
    return codes, scale32, {
        "allzero": False,
        "rotation": rotation_metadata,
        "normalization_eta": eta,
        "norm_squared": norm_squared,
        "quantized_inner_product": denominator,
        "scale_unrounded_float64": scale64,
        "scale_rule": "S=||x||_2^2 / <u,Q(sqrt(d)*u/||x||_2)>; u=x without rotation or u=R(x) with rotation",
        "quantizer_rule": "Q2 on eta*u with boundaries -0.9816, 0, 0.9816 and levels -1.51042, -0.45278, 0.45278, 1.51042",
        "transmitted_scale_dtype": "FP32",
        "rotation_diagnostic_enabled": bool(diagnostics),
        "fp64_reduction": "numpy_elementwise_product_sum",
    }

