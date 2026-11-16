# -*- coding: utf-8 -*-
"""生成 Bellhop 的 ENV、SSP、BTY 和 ATI 输入文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .. import ice_reflection_trc


BOTTOM_GEO_FIELDS = (
    "compressional_speed_mps",
    "shear_speed_mps",
    "density_g_cm3",
    "compressional_attenuation",
    "shear_attenuation",
)


def _write_vector(handle: Any, values: Sequence[float], slash: bool = False) -> None:
    """向 Bellhop 文本文件写入一行数值向量。"""
    array = np.asarray(values).ravel()
    if slash:
        starts = list(range(0, len(array), 8)) or [0]
        for chunk_index, start in enumerate(starts):
            handle.write(" ".join(f"{value:.6f}" for value in array[start:start + 8]))
            handle.write(" /\n" if chunk_index == len(starts) - 1 else "\n")
    else:
        handle.write(" ".join(f"{value:.6f}" for value in array) + "\n")


def _upper_text(value: Any, default: str = "") -> str:
    """把可选值转换为去除首尾空白的大写字符串。"""
    selected = default if value is None or value == "" else value
    return str(selected).strip().upper()


def write_ssp_file(path: str | Path, section: Mapping[str, Any]) -> Path:
    """写出 Bellhop Q 插值使用的距离相关声速剖面文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranges = np.asarray(section["range_km"], dtype=np.float64)
    sound_speed = np.asarray(section["sound_speed_mps"], dtype=np.float64)
    if ranges.ndim != 1 or ranges.size < 3:
        raise ValueError("距离相关 SSP 至少需要 3 个距离点。")
    if sound_speed.ndim != 2 or sound_speed.shape[1] != ranges.size:
        raise ValueError("SSP 声速网格必须是 (depth, range) 形状。")
    if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(sound_speed)):
        raise ValueError("距离相关 SSP 包含非有限数值。")
    if float(ranges[0]) > 1.0e-9 or np.any(np.diff(ranges) <= 0.0):
        raise ValueError("SSP 距离轴必须从 0 km 或更小值开始，并严格递增。")
    with target.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{ranges.size}\n")
        _write_vector(handle, ranges)
        for row in sound_speed:
            _write_vector(handle, row)
    return target


def add_left_guard(section: Mapping[str, Any], guard_km: float = 0.5) -> Dict[str, Any]:
    """在剖面左侧增加负距离保护点，避免 Q 插值在零点附近失败。"""
    ranges = np.asarray(section["range_km"], dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0 or float(ranges[0]) < -1.0e-9:
        return dict(section)
    guard = max(0.001, min(float(guard_km), float(ranges[1] - ranges[0]) * 0.5 if ranges.size > 1 else float(guard_km)))
    guarded: Dict[str, Any] = dict(section)
    guarded["range_km"] = np.concatenate((np.array([-guard], dtype=np.float64), ranges))

    def prepend_first(name: str, axis: int = 0) -> None:
        """给与距离轴对齐的一维或二维字段复制第一个采样点。"""
        if name not in section:
            return
        values = np.asarray(section[name])
        if values.ndim == 1 and values.shape[0] == ranges.size:
            guarded[name] = np.concatenate((values[:1], values), axis=0)
        elif values.ndim == 2 and axis == 1 and values.shape[1] == ranges.size:
            guarded[name] = np.concatenate((values[:, :1], values), axis=1)

    for key in ("bottom_depth_m", "surface_depth_m", "land_mask", "x_km", "y_km"):
        prepend_first(key)
    prepend_first("sound_speed_mps", axis=1)
    bottom_geo = section.get("bottom_geoacoustics")
    if isinstance(bottom_geo, Mapping):
        guarded_geo = dict(bottom_geo)
        fields = bottom_geo.get("fields")
        if isinstance(fields, Mapping):
            guarded_fields: Dict[str, np.ndarray] = {}
            for name, values in fields.items():
                array = np.asarray(values)
                guarded_fields[name] = (
                    np.concatenate((array[:1], array), axis=0)
                    if array.ndim == 1 and array.shape[0] == ranges.size
                    else array
                )
            guarded_geo["fields"] = guarded_fields
        guarded["bottom_geoacoustics"] = guarded_geo
    return guarded


def write_boundary_file(
    path: str | Path,
    ranges_km: Sequence[float],
    depths_m: Sequence[float],
    interpolation: str = "L",
    geoacoustics: Mapping[str, Sequence[float]] | None = None,
) -> Path:
    """写出 Bellhop 短边界文件或包含海底声学参数的长边界文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranges = np.asarray(ranges_km, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    if ranges.shape != depths.shape or ranges.ndim != 1:
        raise ValueError("二维边界的距离和深度必须是一一对应的一维数组。")
    interpolation = _upper_text(interpolation)
    if interpolation not in {"L", "C"}:
        raise ValueError("边界插值方式必须是 L 或 C。")
    long_fields = None
    if geoacoustics is not None:
        long_fields = {}
        for name in BOTTOM_GEO_FIELDS:
            if name not in geoacoustics:
                raise ValueError(f"长格式 BTY 缺少海底声学字段 {name!r}。")
            values = np.asarray(geoacoustics[name], dtype=np.float64)
            if values.shape != ranges.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"长格式 BTY 字段 {name!r} 在每个距离点上都必须是有限数值。")
            long_fields[name] = values
    with target.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"'{interpolation}{'L' if long_fields is not None else ''}'\n")
        handle.write(f"{len(ranges)}\n")
        for index, (range_km, depth_m) in enumerate(zip(ranges, depths)):
            if long_fields is None:
                handle.write(f"{range_km:.6f} {depth_m:.6f}\n")
            else:
                handle.write(
                    f"{range_km:.6f} {depth_m:.6f} "
                    f"{long_fields['compressional_speed_mps'][index]:.6f} "
                    f"{long_fields['shear_speed_mps'][index]:.6f} "
                    f"{long_fields['density_g_cm3'][index]:.6f} "
                    f"{long_fields['compressional_attenuation'][index]:.6f} "
                    f"{long_fields['shear_attenuation'][index]:.6f}\n"
                )
    return target


def write_environment_file(
    path: str | Path,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    environment: Mapping[str, Any],
    section: Mapping[str, Any],
    calculation_radius_km: float,
    receiver_depth_max_m: float,
    run_type: str,
) -> Path:
    """写出支持固定、平均或距离相关 SSP 的 Bellhop 二维 ENV 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model_depth = float(environment["depth_m"][-1])
    source_depth = float(params["source_depth_m"])
    depth_guard_m = float(settings["depth_guard_m"])
    receiver_max = min(float(receiver_depth_max_m), model_depth - depth_guard_m)
    if receiver_max <= settings["receiver_depth_min_m"] or model_depth <= source_depth + depth_guard_m:
        raise ValueError("二维 SSP 没有覆盖声源或接收深度。")
    if run_type.upper() == "R":
        receiver_count, range_count = settings["n_rd"], settings["n_range"]
    else:
        receiver_count, range_count = settings["tl_n_rd"], settings["tl_n_range"]
    ssp_grid = np.asarray(section["sound_speed_mps"], dtype=np.float64)
    expected_shape = (len(environment["depth_m"]), len(section["range_km"]))
    if ssp_grid.shape != expected_shape:
        raise ValueError("距离相关 SSP 的尺寸与深度/距离网格不一致。")
    env_profile = ssp_grid[:, 0] if settings["ssp_mode"] in {"range_dependent", "fixed"} else np.nanmean(ssp_grid, axis=1)
    ray_box_km = calculation_radius_km + float(settings["trace_extra_km"])
    if settings["ssp_mode"] == "range_dependent" and float(section["range_km"][-1]) <= ray_box_km:
        raise ValueError("距离相关 SSP 必须延伸到 Bellhop 声线盒子之外；请增大 BOX_GUARD_KM。")
    fixed_bottom = settings["bottom_fixed"]
    top_option = (
        settings["ssp_interpolation"]
        + settings["surface_boundary"]
        + settings["attenuation_units"]
        + settings["volume_attenuation"]
        + ("*" if settings["use_altimetry_file"] else " ")
    )
    if settings.get("ice_reflection_enabled"):
        top_option = ice_reflection_trc.set_top_boundary_to_trc(top_option)
    bottom_option = settings["bottom_boundary"] + ("*" if settings["use_bathymetry_file"] else " ")
    lines = [
        f"'{settings['title_prefix']} bearing {float(section['bearing_deg']):.6f} deg'",
        f"{float(params['freq_hz']):.3f}",
        str(settings["n_media"]),
        f"'{top_option}'",
        f"{len(environment['depth_m'])} 0.0 {model_depth:.3f}",
    ]
    lines.extend(f"{depth:.3f} {speed:.6f} /" for depth, speed in zip(environment["depth_m"], env_profile))
    lines.extend([
        f"'{bottom_option}' {settings['bottom_roughness_m']:.6f}",
        f"{model_depth:.3f} {fixed_bottom['compressional_speed_mps']:.6f} "
        f"{fixed_bottom['shear_speed_mps']:.6f} {fixed_bottom['density_g_cm3']:.6f} "
        f"{fixed_bottom['compressional_attenuation']:.6f} {fixed_bottom['shear_attenuation']:.6f} /",
        "1",
        f"{source_depth:.3f} /",
        str(receiver_count),
        f"{settings['receiver_depth_min_m']:.3f} {receiver_max:.3f} /",
        str(range_count),
        f"{settings['receiver_range_min_km']:.6f} {calculation_radius_km:.6f} /",
        f"'{run_type.upper()}{settings['beam_type']}'",
        str(settings["n_alpha"]),
        f"{float(params['alpha_min']):.3f} {float(params['alpha_max']):.3f} /",
        f"{settings['step_m']:.3f} {model_depth + settings['zbox_margin_m']:.3f} {ray_box_km:.6f}",
    ])
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target
