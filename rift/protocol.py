"""Public subspaces, joint clipping, and serialized tensor messages."""
from __future__ import annotations
import math
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import torch

CANONICAL_NAMES = (
    "core.base.time_in_day_emb",
    "core.base.day_in_week_emb",
    "core.base.time_series_seasonal.weight",
    "core.base.time_series_seasonal.bias",
    "core.base.time_series_trend.weight",
    "core.base.time_series_trend.bias",
    "core.base.encoder.0.fc.0.weight",
    "core.base.encoder.0.fc.0.bias",
    "core.base.encoder.0.fc.3.weight",
    "core.base.encoder.0.fc.3.bias",
    "core.base.encoder.1.fc.0.weight",
    "core.base.encoder.1.fc.0.bias",
    "core.base.encoder.1.fc.3.weight",
    "core.base.encoder.1.fc.3.bias",
    "core.base.prediction_head.weight",
    "core.base.prediction_head.bias",
    "core.decoder.0.weight",
    "core.decoder.0.bias",
    "core.decoder.2.weight",
    "core.decoder.2.bias",
)
CALENDAR_NAMES = CANONICAL_NAMES[:2]
SHARED_NAMES = tuple(name for name in CANONICAL_NAMES if name not in CALENDAR_NAMES)
CLIP_C = 0.25
DIVISOR = 128
SUM_NOISE_STD = 0.206525
SERVER_ID = 255
SINK_ID = 254
TARGET_ID = 3
MAGIC = b"P24J"
VERSION = 1
KIND_FULL_FP32 = 1
KIND_BASIS_FP32 = 2
KIND_Q2 = 3
KIND_COEFFICIENT_FP32 = 4
FRAME_HEADER = struct.Struct("<4sBBBBBH")
ENTRY_HEADER = struct.Struct("<HBBIIII")
MODE_IDENTITY = 0
MODE_LEFT = 1
MODE_RIGHT = 2



@dataclass(frozen=True)
class TensorBasis:
    """One shared tensor's public matrix view and decoded FP32 basis."""

    name: str
    tensor_shape: tuple[int, ...]
    mode: int
    rows: int
    cols: int
    rank: int
    basis: torch.Tensor | None

    @property
    def coefficient_count(self) -> int:
        if self.mode == MODE_IDENTITY:
            return math.prod(self.tensor_shape)
        if self.mode == MODE_LEFT:
            return self.rank * self.cols
        if self.mode == MODE_RIGHT:
            return self.rows * self.rank
        raise ValueError("basis_mode_invalid")


def _matrix_shape(shape: Sequence[int]) -> tuple[int, int]:
    shape = _shape_tuple(shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("tensor_shape_invalid")
    if len(shape) < 2:
        return 1, math.prod(shape)
    return int(shape[0]), int(math.prod(shape[1:]))


def _identity_basis(name: str, shape: Sequence[int]) -> TensorBasis:
    shape = _shape_tuple(shape)
    rows, cols = _matrix_shape(shape)
    return TensorBasis(name, shape, MODE_IDENTITY, rows, cols, 0, None)


def _shape_tuple(value) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    return tuple(int(dim) for dim in value)


def identity_basis_set(named_shapes: Mapping[str, Sequence[int]],
                       shared_names: Sequence[str] = SHARED_NAMES) -> dict[str, TensorBasis]:
    if tuple(shared_names) != tuple(name for name in CANONICAL_NAMES if name in set(shared_names)):
        raise ValueError("shared_schema_order_invalid")
    return {name: _identity_basis(name, named_shapes[name]) for name in shared_names}


def build_bases_from_global_difference(
    global_difference_fp32: Mapping[str, torch.Tensor],
    shared_names: Sequence[str] = SHARED_NAMES,
) -> dict[str, TensorBasis]:
    """Build causal half-dimensional bases from the actual FP32 predecessor.

    The subtraction that created ``global_difference_fp32`` must already have
    taken place in FP32. SVD is then performed in FP64 and cast to FP32 once.
    """
    if tuple(global_difference_fp32) != tuple(shared_names):
        raise ValueError("global_difference_shared_order_invalid")
    result: dict[str, TensorBasis] = {}
    for name, delta in global_difference_fp32.items():
        if delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all()):
            raise ValueError("global_difference_must_be_finite_fp32")
        shape = tuple(int(x) for x in delta.shape)
        rows, cols = _matrix_shape(shape)
        minimum = min(rows, cols)
        if delta.ndim < 2 or minimum <= 1:
            result[name] = TensorBasis(name, shape, MODE_IDENTITY, rows, cols, 0, None)
            continue
        rank = max(1, minimum // 2)
        matrix32 = delta.detach().cpu().contiguous().reshape(rows, cols)
        if not bool(torch.count_nonzero(matrix32)):
            if rows <= cols:
                basis32 = torch.eye(rows, rank, dtype=torch.float32)
                mode = MODE_LEFT
            else:
                basis32 = torch.eye(cols, rank, dtype=torch.float32)
                mode = MODE_RIGHT
        else:
            matrix64 = matrix32.to(torch.float64)
            left, _singular_values, right_h = torch.linalg.svd(matrix64, full_matrices=False)
            if rows <= cols:
                basis32 = left[:, :rank].to(torch.float32).contiguous()
                mode = MODE_LEFT
            else:
                basis32 = right_h[:rank, :].transpose(0, 1).to(torch.float32).contiguous()
                mode = MODE_RIGHT
        result[name] = TensorBasis(name, shape, mode, rows, cols, rank, basis32)
    return result


def project_tensor(value: torch.Tensor, basis: TensorBasis) -> torch.Tensor:
    """Map one full tensor into its coefficient vector in the received basis."""
    if tuple(value.shape) != basis.tensor_shape:
        raise ValueError("projection_tensor_shape_mismatch")
    if basis.mode == MODE_IDENTITY:
        return value.detach().contiguous().reshape(-1).clone()
    if basis.basis is None or tuple(basis.basis.shape) != (
        (basis.rows, basis.rank) if basis.mode == MODE_LEFT else (basis.cols, basis.rank)
    ):
        raise ValueError("projection_basis_shape_mismatch")
    matrix = value.detach().to(torch.float32).contiguous().reshape(basis.rows, basis.cols)
    b = basis.basis.to(device=matrix.device, dtype=torch.float32)
    if basis.mode == MODE_LEFT:
        projected = b.transpose(0, 1) @ matrix
    elif basis.mode == MODE_RIGHT:
        projected = matrix @ b
    else:
        raise ValueError("projection_mode_invalid")
    return projected.contiguous().reshape(-1)


def lift_coefficients(coefficients: torch.Tensor, basis: TensorBasis) -> torch.Tensor:
    """Lift coefficient values back to the original tensor's full shape."""
    coefficients = coefficients.detach().to(torch.float32).contiguous().reshape(-1)
    if coefficients.numel() != basis.coefficient_count:
        raise ValueError("coefficient_count_mismatch")
    if basis.mode == MODE_IDENTITY:
        return coefficients.reshape(basis.tensor_shape).clone()
    if basis.basis is None:
        raise ValueError("lift_basis_missing")
    b = basis.basis.to(device=coefficients.device, dtype=torch.float32)
    if basis.mode == MODE_LEFT:
        value = b @ coefficients.reshape(basis.rank, basis.cols)
    elif basis.mode == MODE_RIGHT:
        value = coefficients.reshape(basis.rows, basis.rank) @ b.transpose(0, 1)
    else:
        raise ValueError("lift_mode_invalid")
    return value.reshape(basis.tensor_shape).contiguous()


def _coefficient_shape(name: str, named_shapes: Mapping[str, Sequence[int]],
                       bases: Mapping[str, TensorBasis]) -> tuple[int, ...]:
    if name in bases:
        return (bases[name].coefficient_count,)
    return (math.prod(tuple(int(x) for x in named_shapes[name])),)


def map_record_gradients(
    per_record_gradients: Mapping[str, torch.Tensor],
    named_shapes: Mapping[str, Sequence[int]],
    bases: Mapping[str, TensorBasis],
    *,
    present: torch.Tensor | None = None,
    clip_C: float = CLIP_C,
) -> dict[str, torch.Tensor]:
    """Map all20 record gradients, mask absent slots, jointly clip, then sum."""
    if tuple(per_record_gradients) != tuple(named_shapes) or tuple(named_shapes) != CANONICAL_NAMES:
        raise ValueError("per_record_core_schema_mismatch")
    if tuple(bases) != SHARED_NAMES:
        raise ValueError("all_18_shared_bases_required_in_canonical_order")
    first = per_record_gradients[CANONICAL_NAMES[0]]
    if first.ndim < 1:
        raise ValueError("record_gradient_batch_dimension_missing")
    batch = int(first.shape[0])
    if clip_C <= 0:
        raise ValueError("clip_bound_invalid")
    if present is None:
        present = torch.ones(batch, dtype=torch.bool, device=first.device)
    else:
        present = present.to(device=first.device, dtype=torch.bool)
        if tuple(present.shape) != (batch,):
            raise ValueError("presence_mask_shape_invalid")
    mapped: dict[str, torch.Tensor] = {}
    for name in CANONICAL_NAMES:
        value = per_record_gradients[name]
        shape = tuple(int(x) for x in named_shapes[name])
        if value.dtype != torch.float32 or tuple(value.shape) != (batch,) + shape:
            raise ValueError("record_gradient_shape_or_dtype_mismatch")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("record_gradient_nonfinite")
        if name in bases:
            basis = bases[name]
            if tuple(value.shape[1:]) != basis.tensor_shape:
                raise ValueError("record_gradient_basis_shape_mismatch")
            if basis.mode == MODE_IDENTITY:
                mapped[name] = value.detach().flatten(1)
            else:
                matrix = value.detach().reshape(batch, basis.rows, basis.cols)
                b = basis.basis.to(device=matrix.device, dtype=torch.float32)
                coefficients = (b.transpose(0, 1).unsqueeze(0) @ matrix) if basis.mode == MODE_LEFT else (matrix @ b.unsqueeze(0))
                mapped[name] = coefficients.flatten(1)
        else:
            mapped[name] = value.detach().flatten(1)
    if batch == 0:
        return {name: value.new_zeros((value.shape[1],)) for name, value in mapped.items()}
    for name, value in mapped.items():
        mapped[name] = value * present.to(value.dtype).unsqueeze(1)
    joint = torch.cat([mapped[name].double() for name in CANONICAL_NAMES], dim=1)
    norms = joint.norm(dim=1)
    scales = (float(clip_C) / norms.clamp_min(1e-24)).clamp(max=1.0).to(first.dtype)
    return {
        name: (mapped[name] * scales.unsqueeze(1)).sum(dim=0)
        for name in CANONICAL_NAMES
    }


def processed_full_gradients(
    clipped_sums: Mapping[str, torch.Tensor],
    gaussian_sum_noise: Mapping[str, torch.Tensor],
    named_shapes: Mapping[str, Sequence[int]],
    bases: Mapping[str, TensorBasis],
    *,
    divisor: float = DIVISOR,
) -> dict[str, torch.Tensor]:
    """Install the coefficient-domain noisy fixed-divisor mean as 20 full gradients.

    ``gaussian_sum_noise`` is supplied by the caller and is expected to have
    already been sampled at the fixed sum standard deviation. This helper has
    no RNG and cannot inspect sampler membership.
    """
    if tuple(clipped_sums) != CANONICAL_NAMES or tuple(gaussian_sum_noise) != CANONICAL_NAMES:
        raise ValueError("processed_gradient_core_schema_mismatch")
    if tuple(bases) != SHARED_NAMES:
        raise ValueError("all_18_shared_bases_required_in_canonical_order")
    if divisor <= 0:
        raise ValueError("fixed_divisor_invalid")
    output: dict[str, torch.Tensor] = {}
    for name in CANONICAL_NAMES:
        total = clipped_sums[name]
        noise = gaussian_sum_noise[name]
        count = _coefficient_shape(name, named_shapes, bases)[0]
        if total.dtype != torch.float32 or noise.dtype != torch.float32 or total.numel() != count or noise.numel() != count:
            raise ValueError("processed_gradient_noise_shape_or_dtype_mismatch")
        if not bool(torch.isfinite(total).all()) or not bool(torch.isfinite(noise).all()):
            raise ValueError("processed_gradient_nonfinite")
        mean = (total.reshape(-1) + noise.reshape(-1)) / float(divisor)
        if name in bases:
            output[name] = lift_coefficients(mean, bases[name])
        else:
            output[name] = mean.reshape(tuple(named_shapes[name])).contiguous()
    return output


def _numpy_fp32_bytes(value: torch.Tensor) -> bytes:
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError("wire_fp32_tensor_must_be_finite")
    return value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")


def _from_fp32_bytes(payload: bytes, shape: Sequence[int]) -> torch.Tensor:
    count = math.prod(tuple(int(x) for x in shape))
    if len(payload) != count * 4:
        raise ValueError("wire_fp32_payload_length_invalid")
    array = np.frombuffer(payload, dtype="<f4").copy().reshape(tuple(int(x) for x in shape))
    tensor = torch.from_numpy(array)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("wire_fp32_payload_nonfinite")
    return tensor.contiguous()


def _mode_code(mode: int) -> int:
    if mode not in (MODE_IDENTITY, MODE_LEFT, MODE_RIGHT):
        raise ValueError("wire_basis_mode_invalid")
    return int(mode)


def _entry_bytes(index: int, spec: TensorBasis, payload: bytes) -> bytes:
    dims = tuple(int(x) for x in spec.tensor_shape)
    fixed = ENTRY_HEADER.pack(
        int(index), len(dims), _mode_code(spec.mode), int(spec.rows), int(spec.cols),
        int(spec.rank), len(payload),
    )
    return fixed + b"".join(struct.pack("<I", dim) for dim in dims) + payload


def _frame_header(kind: int, round_id: int, sender_id: int, receiver_id: int, entry_count: int) -> bytes:
    if not 0 <= int(round_id) <= 255 or not 0 <= int(sender_id) <= 255 or not 0 <= int(receiver_id) <= 255:
        raise ValueError("wire_identity_out_of_range")
    return FRAME_HEADER.pack(MAGIC, VERSION, int(kind), int(round_id), int(sender_id), int(receiver_id), int(entry_count))


def _parse_frame(payload: bytes, *, kind: int, round_id: int, sender_id: int,
                 receiver_id: int, specs: Mapping[str, TensorBasis]) -> list[bytes]:
    if len(payload) < FRAME_HEADER.size:
        raise ValueError("wire_frame_truncated_header")
    magic, version, observed_kind, observed_round, observed_sender, observed_receiver, entry_count = FRAME_HEADER.unpack_from(payload)
    if (magic, version, observed_kind, observed_round, observed_sender, observed_receiver) != (
        MAGIC, VERSION, int(kind), int(round_id), int(sender_id), int(receiver_id)
    ):
        raise ValueError("wire_frame_identity_or_version_mismatch")
    names = tuple(specs)
    if entry_count != len(names):
        raise ValueError("wire_entry_count_mismatch")
    cursor = FRAME_HEADER.size
    result: list[bytes] = []
    for expected_index, name in enumerate(names):
        if cursor + ENTRY_HEADER.size > len(payload):
            raise ValueError("wire_frame_truncated_entry")
        index, ndim, mode, rows, cols, rank, payload_len = ENTRY_HEADER.unpack_from(payload, cursor)
        cursor += ENTRY_HEADER.size
        spec = specs[name]
        if index != expected_index or ndim != len(spec.tensor_shape):
            raise ValueError("wire_schema_index_or_ndim_mismatch")
        if cursor + 4 * ndim > len(payload):
            raise ValueError("wire_frame_truncated_shape")
        dims = tuple(struct.unpack_from("<I", payload, cursor + 4 * i)[0] for i in range(ndim))
        cursor += 4 * ndim
        if dims != spec.tensor_shape or mode != spec.mode or (rows, cols, rank) != (spec.rows, spec.cols, spec.rank):
            raise ValueError("wire_schema_shape_or_basis_mismatch")
        if cursor + payload_len > len(payload):
            raise ValueError("wire_frame_truncated_payload")
        result.append(payload[cursor:cursor + payload_len])
        cursor += payload_len
    if cursor != len(payload):
        raise ValueError("wire_frame_trailing_bytes")
    return result


def serialize_basis_frame(specs: Mapping[str, TensorBasis], round_id: int, receiver_id: int) -> bytes:
    entries = []
    for index, (name, spec) in enumerate(specs.items()):
        if name != spec.name:
            raise ValueError("basis_name_order_invalid")
        data = b"" if spec.basis is None else _numpy_fp32_bytes(spec.basis)
        expected = 0 if spec.mode == MODE_IDENTITY else (spec.rows * spec.rank if spec.mode == MODE_LEFT else spec.cols * spec.rank) * 4
        if len(data) != expected:
            raise ValueError("basis_wire_payload_size_mismatch")
        entries.append(_entry_bytes(index, spec, data))
    return _frame_header(KIND_BASIS_FP32, round_id, SERVER_ID, receiver_id, len(entries)) + b"".join(entries)


def decode_basis_frame(payload: bytes, expected_specs: Mapping[str, TensorBasis],
                       round_id: int, receiver_id: int) -> dict[str, TensorBasis]:
    parts = _parse_frame(payload, kind=KIND_BASIS_FP32, round_id=round_id,
                         sender_id=SERVER_ID, receiver_id=receiver_id, specs=expected_specs)
    result: dict[str, TensorBasis] = {}
    for (name, spec), data in zip(expected_specs.items(), parts):
        if spec.mode == MODE_IDENTITY:
            if data:
                raise ValueError("identity_basis_has_unexpected_payload")
            basis = None
        else:
            basis_shape = (spec.rows, spec.rank) if spec.mode == MODE_LEFT else (spec.cols, spec.rank)
            basis = _from_fp32_bytes(data, basis_shape)
        result[name] = TensorBasis(name, spec.tensor_shape, spec.mode, spec.rows, spec.cols, spec.rank, basis)
    return result


def _pack_q2(codes: torch.Tensor) -> bytes:
    values = codes.detach().to(torch.int8).cpu().reshape(-1).tolist()
    packed = bytearray((len(values) + 3) // 4)
    for index, item in enumerate(values):
        value = int(item)
        if value not in (-1, 0, 1):
            raise ValueError("q2_code_invalid")
        packed[index // 4] |= (value & 0x3) << ((index % 4) * 2)
    return bytes(packed)


def _unpack_q2(payload: bytes, count: int) -> torch.Tensor:
    if len(payload) != (count + 3) // 4:
        raise ValueError("q2_code_payload_length_invalid")
    values = []
    for index in range(count):
        value = (payload[index // 4] >> ((index % 4) * 2)) & 0x3
        if value == 0x2:
            raise ValueError("q2_reserved_code")
        values.append(value - 4 if value == 0x3 else value)
    if count % 4 and payload and (payload[-1] >> ((count % 4) * 2)) != 0:
        raise ValueError("q2_nonzero_padding")
    return torch.tensor(values, dtype=torch.int8)


def serialize_q2_frame(values: Mapping[str, torch.Tensor], specs: Mapping[str, TensorBasis],
                       round_id: int, sender_id: int) -> tuple[bytes, dict[str, torch.Tensor]]:
    if tuple(values) != tuple(specs):
        raise ValueError("q2_value_order_mismatch")
    entries = []
    decoded: dict[str, torch.Tensor] = {}
    for index, (name, spec) in enumerate(specs.items()):
        raw_value = values[name]
        allowed_shapes = {spec.tensor_shape, (spec.coefficient_count,)}
        if tuple(raw_value.shape) not in allowed_shapes:
            raise ValueError("q2_coefficient_original_shape_mismatch")
        value = raw_value.detach().to(torch.float32).contiguous().reshape(-1)
        if value.numel() != spec.coefficient_count or not bool(torch.isfinite(value).all()):
            raise ValueError("q2_coefficient_shape_or_finiteness_invalid")
        maximum = float(value.abs().max()) if value.numel() else 0.0
        scale = maximum if maximum > 1.0e-12 else 1.0
        codes = torch.zeros_like(value, dtype=torch.int8) if maximum <= 1.0e-12 else torch.round(value / scale).clamp(-1, 1).to(torch.int8)
        packed = _pack_q2(codes)
        scale_bytes = struct.pack("<f", float(scale))
        data = packed + scale_bytes
        entries.append(_entry_bytes(index, spec, data))
        decoded[name] = _unpack_q2(packed, spec.coefficient_count).float() * struct.unpack("<f", scale_bytes)[0]
    frame = _frame_header(KIND_Q2, round_id, sender_id, SERVER_ID, len(entries)) + b"".join(entries)
    return frame, decoded


def decode_q2_frame(payload: bytes, specs: Mapping[str, TensorBasis],
                    round_id: int, sender_id: int) -> dict[str, torch.Tensor]:
    parts = _parse_frame(payload, kind=KIND_Q2, round_id=round_id,
                         sender_id=sender_id, receiver_id=SERVER_ID, specs=specs)
    decoded = {}
    for (name, spec), data in zip(specs.items(), parts):
        count = spec.coefficient_count
        packed_count = (count + 3) // 4
        if len(data) != packed_count + 4:
            raise ValueError("q2_scale_or_code_length_invalid")
        packed, scale_bytes = data[:packed_count], data[packed_count:]
        scale = struct.unpack("<f", scale_bytes)[0]
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("q2_scale_invalid")
        decoded[name] = _unpack_q2(packed, count).float() * scale
    return decoded


def serialize_fp32_frame(values: Mapping[str, torch.Tensor], specs: Mapping[str, TensorBasis],
                         *, kind: int, round_id: int, receiver_id: int) -> bytes:
    if kind not in (KIND_FULL_FP32, KIND_COEFFICIENT_FP32) or tuple(values) != tuple(specs):
        raise ValueError("fp32_frame_kind_or_order_invalid")
    entries = []
    for index, (name, spec) in enumerate(specs.items()):
        value = values[name].detach().to(torch.float32).contiguous()
        expected_shape = spec.tensor_shape if kind == KIND_FULL_FP32 else (spec.coefficient_count,)
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError("fp32_frame_tensor_shape_invalid")
        data = _numpy_fp32_bytes(value)
        entries.append(_entry_bytes(index, spec, data))
    return _frame_header(kind, round_id, SERVER_ID, receiver_id, len(entries)) + b"".join(entries)


def decode_fp32_frame(payload: bytes, specs: Mapping[str, TensorBasis],
                      *, kind: int, round_id: int, receiver_id: int) -> dict[str, torch.Tensor]:
    parts = _parse_frame(payload, kind=kind, round_id=round_id,
                         sender_id=SERVER_ID, receiver_id=receiver_id, specs=specs)
    result = {}
    for (name, spec), data in zip(specs.items(), parts):
        shape = spec.tensor_shape if kind == KIND_FULL_FP32 else (spec.coefficient_count,)
        result[name] = _from_fp32_bytes(data, shape)
    return result


def _validate_full_gradients(named_parameters: Mapping[str, torch.nn.Parameter],
                             gradients: Mapping[str, torch.Tensor]) -> None:
    if tuple(named_parameters) != CANONICAL_NAMES or tuple(gradients) != CANONICAL_NAMES:
        raise ValueError("native_step_schema_invalid")
    for name, parameter in named_parameters.items():
        gradient = gradients[name]
        if tuple(gradient.shape) != tuple(parameter.shape) or gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
            raise ValueError("native_step_gradient_invalid")


def native_adam_step(named_parameters: Mapping[str, torch.nn.Parameter],
                     optimizer: torch.optim.Optimizer,
                     gradients: Mapping[str, torch.Tensor],
                     expected_step: int) -> dict[str, torch.Tensor]:
    """Install processed full gradients, then step the persistent native Adam."""
    _validate_full_gradients(named_parameters, gradients)
    bound = optimizer.param_groups[0]["params"]
    if len(bound) != len(CANONICAL_NAMES) or any(id(a) != id(b) for a, b in zip(bound, named_parameters.values())):
        raise ValueError("native_adam_parameter_binding_changed")
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in named_parameters.items():
        parameter.grad = gradients[name].detach().clone()
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in named_parameters.values()):
        raise FloatingPointError("native_adam_parameter_nonfinite")
    if not all(int(optimizer.state[parameter]["step"].item()) == expected_step for parameter in named_parameters.values()):
        raise ValueError("native_adam_clock_mismatch")
    result = {name: parameter.detach().clone() for name, parameter in named_parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def projected_displacement_step(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    full_gradients: Mapping[str, torch.Tensor],
    shared_reference: Mapping[str, torch.Tensor],
    bases: Mapping[str, TensorBasis],
    accumulated_coefficients: Mapping[str, torch.Tensor],
    expected_step: int,
) -> dict[str, torch.Tensor]:
    """Step native Adam, project each shared displacement, and reinstall cumulative lift."""
    if tuple(bases) != SHARED_NAMES or tuple(shared_reference) != SHARED_NAMES or tuple(accumulated_coefficients) != SHARED_NAMES:
        raise ValueError("projected_displacement_schema_invalid")
    before = {name: named_parameters[name].detach().clone() for name in bases}
    native_adam_step(named_parameters, optimizer, full_gradients, expected_step)
    updated_accum: dict[str, torch.Tensor] = {}
    for name, spec in bases.items():
        displacement = named_parameters[name].detach() - before[name]
        projected = project_tensor(displacement, spec)
        updated = accumulated_coefficients[name].detach().to(torch.float32).reshape(-1) + projected
        updated_accum[name] = updated
        reconstructed = shared_reference[name].detach().to(torch.float32) + lift_coefficients(updated, spec)
        named_parameters[name].copy_(reconstructed)
    return updated_accum

