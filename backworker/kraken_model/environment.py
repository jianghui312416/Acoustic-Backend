# -*- coding: utf-8 -*-
"""生成 Kraken 环境文件和 FIELD 声场参数文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _safe_title(value: Any) -> str:
    """生成仅含 ASCII 的安全标题，兼容旧版 Fortran 求解器。"""
    cleaned = str(value or "Kraken calculation").replace("'", " ").strip()
    ascii_title = cleaned.encode("ascii", errors="replace").decode("ascii")
    ascii_title = " ".join(ascii_title.split())
    return ascii_title or "Kraken calculation"


def _number_line(values: Iterable[Any], suffix: str = "/") -> str:
    """把一组数值写成 Acoustics Toolbox 可读取的一行。"""
    body = " ".join(f"{float(value):.10g}" for value in values)
    return f"{body} {suffix}".rstrip()


def _counted_vector(values: Sequence[Any]) -> str:
    """把向量写成“数量行 + 数值行”格式，兼容 Fortran 的逐记录读取。"""
    return f"{len(values)}\n{_number_line(values)}"


def receiver_ranges(config: Mapping[str, Any]) -> np.ndarray:
    """根据 FIELD 配置生成等间距接收距离向量，单位为千米。"""
    spec = config["FIELD"]["RECEIVER_RANGES_KM"]
    return np.linspace(
        float(spec["MIN"]),
        float(spec["MAX"]),
        int(spec.get("COUNT", config["TL_N_RANGE"])),
        dtype=np.float64,
    )


def write_environment_file(path: str | Path, config: Mapping[str, Any]) -> Path:
    """写出可由 Kraken 直接读取的范围无关 `.env` 环境文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    environment = config["ENVIRONMENT"]
    field = config["FIELD"]
    bottom = environment["BOTTOM"]
    options = "".join([
        str(environment.get("INTERPOLATION", "C")).upper(),
        str(environment.get("TOP_BOUNDARY", "V")).upper(),
        str(environment.get("ATTENUATION_UNITS", "W")).upper(),
    ])
    water_density = float(environment.get("WATER_DENSITY_G_CM3", 1.0))
    water_attenuation = float(environment.get("WATER_ATTENUATION", 0.0))
    lines = [
        f"'{_safe_title(environment.get('TITLE'))}'",
        f"{float(environment['FREQUENCY_HZ']):.10g}",
        "1",
        f"'{options}'",
    ]
    if str(environment.get("TOP_BOUNDARY", "V")).upper() in {"S", "H", "T", "I"}:
        twersky = environment["TWERSKY"]
        lines.append(
            f"{float(twersky['BUMDEN']):.10g} {float(twersky['ETA_M']):.10g} {float(twersky['XI_M']):.10g}"
        )
    lines.append(
        (
            f"{int(environment.get('MESH_POINTS', 0))} "
            f"{float(environment.get('INTERFACE_ROUGHNESS_M', 0.0)):.10g} "
            f"{float(environment['WATER_DEPTH_M']):.10g}"
        )
    )
    for depth, speed, *_ in environment["SOUND_SPEED_PROFILE"]:
        lines.append(
            f"{float(depth):.10g} {float(speed):.10g} 0.0 "
            f"{water_density:.10g} {water_attenuation:.10g} 0.0"
        )
    bottom_condition = str(bottom.get("BOUNDARY_CONDITION", "A")).upper()
    lines.append(f"'{bottom_condition}' {float(bottom.get('ROUGHNESS_M', 0.0)):.10g}")
    if bottom_condition == "A":
        lines.append(
            f"{float(environment['WATER_DEPTH_M']):.10g} "
            f"{float(bottom['COMPRESSIONAL_SPEED_MPS']):.10g} "
            f"{float(bottom.get('SHEAR_SPEED_MPS', 0.0)):.10g} "
            f"{float(bottom['DENSITY_G_CM3']):.10g} "
            f"{float(bottom.get('COMPRESSIONAL_ATTENUATION', 0.0)):.10g} "
            f"{float(bottom.get('SHEAR_ATTENUATION', 0.0)):.10g}"
        )
    lines.extend([
        (
            f"{float(environment.get('PHASE_SPEED_MIN_MPS', 0.0)):.10g} "
            f"{float(environment['PHASE_SPEED_MAX_MPS']):.10g}"
        ),
        f"{float(environment['MAX_RANGE_KM']):.10g}",
        _counted_vector(field["SOURCE_DEPTHS_M"]),
        _counted_vector(field["RECEIVER_DEPTHS_M"]),
    ])
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def _profile_ranges(values: Sequence[Any] | None) -> np.ndarray:
    """校验 FIELD 所需的环境剖面起始距离，单位为千米。"""
    ranges = np.asarray([0.0] if values is None else values, dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0 or not np.all(np.isfinite(ranges)):
        raise ValueError("Kraken 环境剖面距离必须是非空的有限一维数组。")
    if abs(float(ranges[0])) > 1.0e-9:
        raise ValueError("Kraken 第一个环境剖面必须从 0 km 开始。")
    if np.any(np.diff(ranges) <= 0.0):
        raise ValueError("Kraken 环境剖面距离必须严格递增。")
    return ranges


def write_field_file(
    path: str | Path,
    config: Mapping[str, Any],
    profile_ranges_km: Sequence[Any] | None = None,
) -> Path:
    """写出 FIELD 的 `.flp` 文件，并声明所有沿程环境剖面。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    field = config["FIELD"]
    ranges = receiver_ranges(config)
    receiver_depths = field["RECEIVER_DEPTHS_M"]
    options = f"{str(field.get('SOURCE_TYPE', 'R')).upper()}{str(field.get('MODE_THEORY', 'A')).upper()}"
    range_spec = f"{len(ranges)}\n{ranges[0]:.10g} {ranges[-1]:.10g} /"
    profiles = _profile_ranges(profile_ranges_km)
    zero_displacements = [0.0] * len(receiver_depths)
    lines = [
        "/",
        f"'{options}'",
        str(int(field.get("MAX_MODES", 9999))),
        f"{len(profiles)}\n{_number_line(profiles)}",
        range_spec,
        _counted_vector(field["SOURCE_DEPTHS_M"]),
        _counted_vector(receiver_depths),
        _counted_vector(zero_displacements),
    ]
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target
