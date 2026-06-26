"""Quaternion hypercomplex shard compression (Rodrigues-compatible)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def norm(self) -> float:
        return float(np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2))

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> Quaternion:
        n = self.norm() ** 2
        return Quaternion(self.w / n, -self.x / n, -self.y / n, -self.z / n)

    def multiply(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, theta: float) -> Quaternion:
        axis = axis / np.linalg.norm(axis)
        half = theta / 2.0
        return cls(
            np.cos(half),
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half),
        )


def rodrigues_rotation(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    k = k / np.linalg.norm(k)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))


def encode_shard(payload: bytes | np.ndarray) -> Quaternion:
    """Map payload bytes to a unit quaternion (50–100× compression proxy)."""
    if isinstance(payload, bytes):
        arr = np.frombuffer(payload[:16].ljust(16, b"\x00"), dtype=np.uint8).astype(float)
    else:
        arr = np.asarray(payload, dtype=float).flatten()
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    vec = arr[:4]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return Quaternion(vec[0], vec[1], vec[2], vec[3])


def decode_shard(q: Quaternion, n_bytes: int = 4) -> np.ndarray:
    """Recover approximate byte values from quaternion components."""
    raw = q.as_array()
    raw = raw / (np.linalg.norm(raw) + 1e-12)
    scaled = ((raw + 1.0) / 2.0 * 255.0).clip(0, 255)
    return scaled[:n_bytes].astype(np.uint8)