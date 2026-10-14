"""Stateful predictive Q2 uploads and affine Q4 model synchronization."""
from __future__ import annotations

import struct
import numpy as np
import torch

from . import protocol as p
from . import source_wire


def clone(state):
    return {k: v.detach().cpu().float().clone() for k, v in state.items()}


class PredictiveEndpoint:
    """One endpoint's history. Sender and receiver must use separate instances."""

    def __init__(self):
        self.history = None
        self.next_round = 0

    def prediction(self, specs):
        if self.history is None:
            return {k: torch.zeros(s.coefficient_count) for k, s in specs.items()}
        return {k: p.project_tensor(self.history[k], s) for k, s in specs.items()}

    def encode(self, coefficients, specs, round_id, source_id):
        prediction = self.prediction(specs)
        residual = {k: coefficients[k].detach().cpu().reshape(-1) - prediction[k] for k in specs}
        payload, _ = source_wire._serialize(p, 'gaussian4_rotated', residual, specs, round_id, source_id)
        # The sender also advances from the actual serialized representation.
        self.decode(payload, specs, round_id, source_id)
        return payload

    def decode(self, payload, specs, round_id, source_id):
        if round_id != self.next_round:
            raise ValueError('Predictive messages must be accepted exactly once, in round order.')
        prediction = self.prediction(specs)
        residual = source_wire._decode(p, 'gaussian4_rotated', payload, specs, round_id, source_id)
        decoded = {k: prediction[k] + residual[k] for k in specs}
        innovation = {k: p.lift_coefficients(decoded[k] - prediction[k], s) for k, s in specs.items()}
        update = {k: p.lift_coefficients(decoded[k], s) for k, s in specs.items()}
        self.history = clone(update)
        self.next_round += 1
        return update, innovation


def affine_encode(candidate, reference, specs, round_id, receiver_id, bits=4):
    """Quantize increments, not absolute parameters; keep min/scale in FP32."""
    if bits != 4:
        raise ValueError('This release implements the evaluated affine Q4 broadcast.')
    entries = []
    for i, (name, spec) in enumerate(specs.items()):
        x = (candidate[name].cpu().float() - reference[name].cpu().float()).reshape(-1).numpy()
        if not np.isfinite(x).all():
            raise FloatingPointError('Nonfinite broadcast increment.')
        lo, hi = np.float32(x.min()), np.float32(x.max())
        scale = np.float32((np.float64(hi) - np.float64(lo)) / 15.0)
        if scale == 0:
            codes = np.zeros(x.size, dtype=np.uint8)
        else:
            codes = np.rint((x.astype(np.float64) - lo) / scale).clip(0, 15).astype(np.uint8)
        packed = np.zeros((codes.size + 1) // 2, dtype=np.uint8)
        packed[:] = codes[::2]
        packed[:codes[1::2].size] |= codes[1::2] << 4
        entries.append(p._entry_bytes(i, spec, struct.pack('<ff', scale, lo) + packed.tobytes()))
    return p._frame_header(44, round_id, p.SERVER_ID, receiver_id, len(entries)) + b''.join(entries)


def affine_decode(payload, reference, specs, round_id, receiver_id):
    parts = p._parse_frame(payload, kind=44, round_id=round_id, sender_id=p.SERVER_ID,
                           receiver_id=receiver_id, specs=specs)
    result = {}
    for (name, spec), data in zip(specs.items(), parts):
        count = spec.coefficient_count
        if len(data) != 8 + (count + 1) // 2:
            raise ValueError('Invalid affine payload length.')
        scale, lo = struct.unpack('<ff', data[:8])
        if not np.isfinite([scale, lo]).all() or scale < 0:
            raise ValueError('Invalid affine scale/minimum.')
        packed = np.frombuffer(data[8:], dtype=np.uint8)
        codes = np.empty(packed.size * 2, dtype=np.uint8)
        codes[::2], codes[1::2] = packed & 15, packed >> 4
        delta = np.float32(lo) + np.float32(scale) * codes[:count].astype(np.float32)
        result[name] = reference[name].cpu().float() + torch.from_numpy(delta.copy()).reshape(spec.tensor_shape)
    return result


class CommunicationLedger:
    """Serialized application bytes, counted once per point-to-point delivery."""

    def __init__(self):
        self.messages = []

    def add(self, direction, category, round_id, peer, payload):
        self.messages.append(dict(direction=direction, category=category, round=round_id,
                                  peer=peer, bytes=len(payload)))

    def summary(self):
        totals = {d: sum(x['bytes'] for x in self.messages if x['direction'] == d)
                  for d in ('upload', 'download')}
        totals['total'] = sum(totals.values())
        return {'bytes': totals, 'MB': {k: v / 1e6 for k, v in totals.items()},
                'categories_bytes': {c: sum(x['bytes'] for x in self.messages if x['category'] == c)
                                     for c in sorted({x['category'] for x in self.messages})},
                'messages': self.messages}
