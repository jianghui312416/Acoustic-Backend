"""由 Bellhop 垂直阵复声压构造环境噪声互谱和垂直指向性。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..bellhop_model.results import read_shd_bin
from .wind_noise_receiver import (
    RAIN_MODEL_MAX_FREQUENCY_HZ,
    RAIN_MODEL_MIN_FREQUENCY_HZ,
    rain_cell_source_psd_linear,
    rain_surface_source_depth_m,
    shipping_cell_source_power_linear,
    wind_cell_source_psd_linear,
)


_BEARING_RE = re.compile(r"bearing_\d+_(\d+)p(\d+)deg", re.IGNORECASE)


def _bearing_from_path(path: Path) -> float:
    match = _BEARING_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"无法从 Bellhop 方位目录读取角度：{path.parent.name}")
    return float(f"{match.group(1)}.{match.group(2)}") % 360.0


def _load_bearing_fields(job_dir: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(job_dir).glob("sources/source_*/bearings_2d/bearing_*/Point2D_TL.shd"))
    if not paths:
        raise FileNotFoundError(f"Bellhop 阵元任务缺少 Point2D_TL.shd：{job_dir}")
    fields = []
    for path in paths:
        result = read_shd_bin(path)
        pressure = np.asarray(result["pressure"], dtype=np.complex128)
        if pressure.shape[0:2] != (1, 1):
            raise ValueError(f"垂直阵 Bellhop SHD 维度异常：{pressure.shape}")
        fields.append({
            "bearing_deg": _bearing_from_path(path),
            "depth_m": np.asarray(result["receiver_depth_m"], dtype=np.float64),
            "range_km": np.asarray(result["range_km"], dtype=np.float64),
            "pressure": pressure[0, 0],
        })
    return fields


def _depth_slice(field: Mapping[str, Any], depth_m: float) -> np.ndarray:
    depth = np.asarray(field["depth_m"], dtype=np.float64)
    pressure = np.asarray(field["pressure"], dtype=np.complex128)
    if pressure.shape[0] != depth.size:
        raise ValueError("Bellhop 复声压深度轴不一致。")
    wanted = float(depth_m)
    if wanted < float(depth[0]) - 1.0e-6 or wanted > float(depth[-1]) + 1.0e-6:
        return np.full(pressure.shape[1], np.nan + 1j * np.nan)
    upper = int(np.searchsorted(depth, wanted, side="left"))
    if upper <= 0:
        return pressure[0]
    if upper >= depth.size:
        return pressure[-1]
    if abs(float(depth[upper]) - wanted) <= 1.0e-6:
        return pressure[upper]
    lower = upper - 1
    weight = (wanted - depth[lower]) / (depth[upper] - depth[lower])
    return pressure[lower] * (1.0 - weight) + pressure[upper] * weight


def _sample_selection_transfer(
    fields: Sequence[Mapping[str, Any]],
    selection: Any,
    source_depth_m: float,
) -> np.ndarray:
    field_bearings = np.asarray([item["bearing_deg"] for item in fields], dtype=np.float64)
    source_bearings = np.asarray(selection.bearing_deg, dtype=np.float64)
    delta = np.abs(
        (field_bearings[:, None] - source_bearings[None, :] + 180.0) % 360.0 - 180.0
    )
    field_index = np.argmin(delta, axis=0)
    result = np.full(selection.source_count, np.nan + 1j * np.nan, dtype=np.complex128)
    distances = np.asarray(selection.distance_km, dtype=np.float64)
    for selected_index in np.unique(field_index):
        indexes = np.flatnonzero(field_index == selected_index)
        field = fields[int(selected_index)]
        ranges = np.asarray(field["range_km"], dtype=np.float64)
        values = _depth_slice(field, source_depth_m)
        good = np.isfinite(ranges) & np.isfinite(values.real) & np.isfinite(values.imag)
        if np.count_nonzero(good) < 2:
            continue
        order = np.argsort(ranges[good])
        x = ranges[good][order]
        y = values[good][order]
        requested = distances[indexes]
        inside = (requested >= x[0] - 1.0e-7) & (requested <= x[-1] + 1.0e-7)
        if not np.any(inside):
            continue
        target = indexes[inside]
        result[target] = (
            np.interp(requested[inside], x, y.real)
            + 1j * np.interp(requested[inside], x, y.imag)
        )
    return result


def beamform_cross_spectral_matrix(
    cross_spectral_matrix: Any,
    element_depth_m: Any,
    frequency_hz: float,
    scan_angle_deg: Any,
    sound_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按论文式(9)计算常规波束形成，并返回绝对级与峰值归一化级。"""
    csd = np.asarray(cross_spectral_matrix, dtype=np.complex128)
    depth = np.asarray(element_depth_m, dtype=np.float64).reshape(-1)
    angle = np.asarray(scan_angle_deg, dtype=np.float64).reshape(-1)
    if csd.shape != (depth.size, depth.size):
        raise ValueError("互谱密度矩阵尺寸必须与有效阵元数一致。")
    if depth.size < 2 or angle.size < 2:
        raise ValueError("垂直阵至少需要两个有效阵元和两个扫描角。")
    if not np.allclose(csd, csd.conj().T, rtol=1.0e-7, atol=1.0e-10):
        raise ValueError("互谱密度矩阵必须为 Hermitian 矩阵。")
    frequency = float(frequency_hz)
    sound_speed = float(sound_speed_mps)
    if not (math.isfinite(frequency) and frequency > 0.0):
        raise ValueError("波束形成频率必须为正有限数。")
    if not (math.isfinite(sound_speed) and sound_speed > 0.0):
        raise ValueError("波束形成声速必须为正有限数。")

    relative_depth = depth - depth[0]
    wave_number = 2.0 * math.pi * frequency / sound_speed
    steering = np.exp(
        -1j
        * wave_number
        * np.sin(np.deg2rad(angle))[:, None]
        * relative_depth[None, :]
    )
    beam_power = np.real(np.einsum("ai,ij,aj->a", steering.conj(), csd, steering))
    beam_power = np.maximum(beam_power, np.finfo(np.float64).tiny)
    normalized_db = 10.0 * np.log10(beam_power / float(np.max(beam_power)))
    # CSD 已由源功率与 Bellhop 复传递函数构成，q^H C q 因而就是
    # 未归一化的阵列输出功率；同时返回论文绘图所用的峰值归一化级。
    absolute_db = 10.0 * np.log10(beam_power)
    return absolute_db, normalized_db


def source_grid_vertical_directivity(
    frequency_hz: float,
    selections: Mapping[str, Any],
    element_job_dirs: Sequence[str | Path],
    element_depth_m: Any,
    scan_angle_deg: Any,
    *,
    active_element_indexes: Sequence[int],
    sound_speed_mps: float = 1500.0,
    ice_concentration: float = 0.0,
) -> dict[str, Any]:
    """由独立网格源的 Bellhop 复声压构造 CSD，并计算垂直指向性。"""
    frequency = float(frequency_hz)
    depths = np.asarray(element_depth_m, dtype=np.float64)
    if len(element_job_dirs) != depths.size:
        raise ValueError("Bellhop 阵元任务数必须等于垂直阵阵元数。")
    active = np.asarray(active_element_indexes, dtype=int)
    if active.ndim != 1 or active.size < 2 or np.any(active < 0) or np.any(active >= depths.size):
        raise ValueError("有效阵元索引无效。")
    fields_by_element = [_load_bearing_fields(path) for path in element_job_dirs]
    csd = np.zeros((active.size, active.size), dtype=np.complex128)
    component_counts: dict[str, int] = {}
    open_fraction = float(np.clip(1.0 - ice_concentration, 0.0, 1.0))

    component_specs = []
    if "wind" in selections:
        component_specs.append((
            "wind",
            selections["wind"],
            float(selections["wind"].source_depth_m),
            wind_cell_source_psd_linear(selections["wind"], frequency) * open_fraction,
        ))
    if "rain" in selections and RAIN_MODEL_MIN_FREQUENCY_HZ <= frequency <= RAIN_MODEL_MAX_FREQUENCY_HZ:
        component_specs.append((
            "rain",
            selections["rain"],
            rain_surface_source_depth_m(frequency),
            rain_cell_source_psd_linear(selections["rain"], frequency) * open_fraction,
        ))
    if "shipping" in selections:
        component_specs.append((
            "shipping",
            selections["shipping"],
            float(selections["shipping"].source_depth_m),
            shipping_cell_source_power_linear(selections["shipping"], frequency),
        ))

    for name, selection, source_depth, source_power in component_specs:
        transfers = np.vstack([
            _sample_selection_transfer(fields_by_element[int(index)], selection, source_depth)
            for index in active
        ])
        power = np.asarray(source_power, dtype=np.float64)
        valid = (
            np.isfinite(power)
            & (power > 0.0)
            & np.all(np.isfinite(transfers.real) & np.isfinite(transfers.imag), axis=0)
        )
        component_counts[name] = int(np.count_nonzero(valid))
        if not np.any(valid):
            continue
        weighted = transfers[:, valid] * np.sqrt(power[valid])[None, :]
        csd += weighted @ weighted.conj().T

    if not np.any(np.real(np.diag(csd)) > 0.0):
        raise ValueError("Bellhop 垂直阵互谱矩阵没有有效网格源贡献。")
    absolute_db, normalized_db = beamform_cross_spectral_matrix(
        csd,
        depths[active],
        frequency,
        scan_angle_deg,
        sound_speed_mps,
    )
    return {
        "grazingAngle": np.asarray(scan_angle_deg, dtype=np.float64),
        "absNoiseLevel": absolute_db,
        "normalizedLevelDb": normalized_db,
        "crossSpectralMatrix": csd,
        "activeElementIndexes": active,
        "componentSourceCounts": component_counts,
    }
