#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为冰覆盖区域生成 Bellhop/KRAKEN 顶边界 .trc 反射表。

这个模块把 V6 版冰下反射流程整理成后端可复用函数：

1. 按顶部入口选择 HYCOM/CICE、neXtSIM 或 ICESat-2 IS2SITMOGR4 海冰数据。
2. 以声源为中心选取计算区域，得到一个代表性 BUMDEN_domain。
3. 使用混合 Burke-Twersky 风格反射模型：
   低/中频使用数值 Mathieu 级数；
   高频使用数值 Kirchhoff / 物理光学分支。
4. 输出 Bellhop 或 KRAKEN ``F`` 顶边界需要的 .trc 文件。

默认不额外生成图、CSV、ENV 或 JSON 附属文件。
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


FILL_LIMIT = -900.0
EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# 冰反射数据源入口（类似宏定义，在这里统一切换）
# =============================================================================
# 可选值：
#   "HYCOM"  ：使用 HYCOM/CICE 的 sic、sih 数据做参数化估算。
#   "NEXTSIM"：使用 neXtSIM 的 siconc、sithick 数据做同一参数化估算。
#   "ICESAT2"：使用 IS2SITMOGR4 的冰厚、干舷、积雪等数据推算冰吃水与 BUMDEN。
#
# HYCOM 和 neXtSIM 的平均冰厚都不直接观测冰底冰脊，得到的是“等效 BUMDEN”。
# ICESat-2 月度格网同样不能视为某一小时的冰脊实测。一次只选择一种来源。
ICE_REFLECTION_DATA_SOURCE = "HYCOM"

# HYCOM 参数化常量也供 neXtSIM 模式复用；两种模型文件使用不同变量名。
# 若后续用同期观测完成标定，可调整 BUMDEN_SCALE，不需改 BT 或 .trc 写出逻辑。
# 数据库模式保持为 None：使用数据路由按 dataSource、dataType 和请求时间
# 返回的 HYCOM 文件。只有离线调试时才在这里临时填写一个本地 Path。
HYCOM_ICE_DATA_FILE: Path | None = None
# neXtSIM 和 ICESat-2 仅保留为离线算法兼容入口；当前门户统一使用
# HYCOM 服务按前端所选时间获取海冰数据。
NEXTSIM_ICE_DATA_FILE: Path | None = None
ICESAT2_ICE_DATA_FILE: Path | None = None
HYCOM_SIC_VARIABLE = "sic"
HYCOM_SIH_VARIABLE = "sih"
HYCOM_LAT_VARIABLE = "lat"
HYCOM_LON_VARIABLE = "lon"
HYCOM_TIME_VARIABLE = "time"
HYCOM_TIME_INDEX = 0
HYCOM_ICE_DENSITY_KG_M3 = 917.0
HYCOM_WATER_DENSITY_KG_M3 = 1025.0
HYCOM_WEIGHT_BUMDEN_BY_SIC = True
HYCOM_BUMDEN_SCALE = 1.0
NEXTSIM_SIC_VARIABLE = "siconc"
NEXTSIM_SIH_VARIABLE = "sithick"
NEXTSIM_LAT_VARIABLE = "latitude"
NEXTSIM_LON_VARIABLE = "longitude"


@dataclass(frozen=True) # frozen=true 冻结实例，防止修改
class GeometryParams:
    eta_m: float = 8.2
    xi_m: float = 5.1
    sound_speed_m_s: float = 1500.0
    residual_tol: float = 1e-4

    def validate(self) -> None:
        if self.eta_m <= 0.0 or self.xi_m <= 0.0:
            raise ValueError("ETA 和 XI 必须大于 0")
        if self.xi_m >= self.eta_m:
            raise ValueError("在半椭圆模型中，XI 必须小于 ETA")
        if self.sound_speed_m_s <= 0.0:
            raise ValueError("sound_speed_m_s 必须大于 0")


@dataclass(frozen=True)
class HFConfig:
    points_per_wavelength: float = 32.0
    convergence_points_per_wavelength: float = 16.0
    min_boundary_points: int = 1024
    recommended_kmin: float = 10.0

    def validate(self) -> None:
        if self.points_per_wavelength < 8.0:
            raise ValueError("points_per_wavelength 建议不小于 8")
        if self.convergence_points_per_wavelength <= 0.0:
            raise ValueError("convergence_points_per_wavelength 必须大于 0")
        if self.min_boundary_points < 256:
            raise ValueError("min_boundary_points 建议不小于 256")


@dataclass(frozen=True)
class HybridConfig:
    blend_kb_start: float = 16.0
    blend_kb_end: float = 24.0
    hf_config: HFConfig = field(default_factory=HFConfig)

    def validate(self) -> None:
        if self.blend_kb_start <= 0.0:
            raise ValueError("blend_kb_start 必须大于 0")
        if self.blend_kb_end <= self.blend_kb_start:
            raise ValueError("blend_kb_end 必须大于 blend_kb_start")


@dataclass(frozen=True)
class IceProcessingConfig:
    window_cells: int = 5
    background_percentile: float = 25.0
    min_window_valid: int = 5
    min_sic: float = 0.15
    min_segments: int = 1


@dataclass(frozen=True)
class HycomProcessingConfig:
    """HYCOM 平均冰厚到等效 BUMDEN 的参数化设置。"""

    window_cells: int = 5
    background_percentile: float = 25.0
    min_window_valid: int = 5
    min_sic: float = 0.15
    time_index: int = HYCOM_TIME_INDEX
    ice_density_kg_m3: float = HYCOM_ICE_DENSITY_KG_M3
    water_density_kg_m3: float = HYCOM_WATER_DENSITY_KG_M3
    weight_by_sic: bool = HYCOM_WEIGHT_BUMDEN_BY_SIC
    bumden_scale: float = HYCOM_BUMDEN_SCALE
    sic_variable: str = HYCOM_SIC_VARIABLE
    sih_variable: str = HYCOM_SIH_VARIABLE
    lat_variable: str = HYCOM_LAT_VARIABLE
    lon_variable: str = HYCOM_LON_VARIABLE
    time_variable: str = HYCOM_TIME_VARIABLE

    def validate(self) -> None:
        if self.window_cells < 3 or self.window_cells % 2 != 1:
            raise ValueError("HYCOM window_cells 必须是不小于 3 的奇数")
        if not 0.0 <= self.background_percentile <= 100.0:
            raise ValueError("HYCOM background_percentile 必须在 0 到 100 之间")
        if not 1 <= self.min_window_valid <= self.window_cells**2:
            raise ValueError("HYCOM min_window_valid 必须在窗口网格数范围内")
        if not 0.0 <= self.min_sic <= 1.0:
            raise ValueError("HYCOM min_sic 必须在 0 到 1 之间")
        if self.time_index < 0:
            raise ValueError("HYCOM time_index 必须大于等于 0")
        if self.ice_density_kg_m3 <= 0.0 or self.water_density_kg_m3 <= 0.0:
            raise ValueError("HYCOM 冰和海水密度必须大于 0")
        if self.bumden_scale < 0.0:
            raise ValueError("HYCOM bumden_scale 必须大于等于 0")


@dataclass(frozen=True)
class DomainConfig:
    radius_km: float
    aggregate: str = "median"
    min_domain_cells: int = 3


@dataclass(frozen=True)
class TrcGenerationResult:
    path: Path
    frequency_hz: float
    bumden_domain_ridges_per_km: float
    selected_model: str
    solver_valid: bool
    angle_count: int
    passivity_clipped_count: int
    diagnostics: Mapping[str, Any]


def angle_grid() -> np.ndarray:
    """返回 V6 版 Bellhop 使用的掠射角网格，单位为度。"""
    return np.unique(
        np.concatenate(
            [
                np.arange(0.5, 10.0, 0.25),
                np.arange(10.0, 30.0, 0.5),
                np.arange(30.0, 89.5 + 1e-12, 1.0),
            ]
        )
    ).astype(float)


def freq_tag(frequency_hz: float) -> str:
    """把频率数值整理成适合文件名使用的标签。"""
    return f"{float(frequency_hz):g}".replace(".", "p")


def write_trc(path: str | Path, grazing_angle_deg: Sequence[float], reflection: Sequence[complex]) -> Path:
    """写出 Bellhop/KRAKEN 顶边界反射系数表。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(grazing_angle_deg, dtype=float)
    r = np.asarray(reflection, dtype=np.complex128)
    if theta.ndim != 1 or r.ndim != 1 or theta.size != r.size:
        raise ValueError("grazing_angle_deg 和 reflection 必须是一一对应的一维数组")
    if theta.size < 2:
        raise ValueError("Bellhop .trc 至少需要两个角度点")
    magnitude = np.abs(r)
    phase_deg = np.rad2deg(np.unwrap(np.angle(r)))
    with out.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{theta.size}\n")
        for theta_i, mag_i, phase_i in zip(theta, magnitude, phase_deg):
            handle.write(f"{theta_i:11.6f}  {mag_i:14.10f}  {phase_i:14.8f}\n")
    return out


def cover_full_grazing_angle_domain(
    grazing_angle_deg: Sequence[float],
    reflection: Sequence[complex],
) -> tuple[np.ndarray, np.ndarray]:
    """用最近内点的极限值补齐 KRAKEN 需要的 0°和 90°。

    Burke–Twersky 数值分支只允许严格位于 ``(0, 90)`` 的角度；
    KRAKEN 在求根过程中又会查询端点。因此先在内点求解，
    再以最近内点的复反射系数作为端点极限，避免求解器把
    表外反射系数设为零。
    """
    theta = np.asarray(grazing_angle_deg, dtype=float)
    values = np.asarray(reflection, dtype=np.complex128)
    if theta.ndim != 1 or values.ndim != 1 or theta.size != values.size:
        raise ValueError("grazing_angle_deg 和 reflection 必须是一一对应的一维数组")
    if theta.size < 2 or np.any(~np.isfinite(theta)):
        raise ValueError("掠射角至少需要两个有限内点")
    if np.any(np.diff(theta) <= 0.0) or theta[0] <= 0.0 or theta[-1] >= 90.0:
        raise ValueError("KRAKEN 反射表的求解角度必须在 0° 到 90°之间严格递增")
    return (
        np.concatenate(([0.0], theta, [90.0])),
        np.concatenate(([values[0]], values, [values[-1]])),
    )


def set_top_boundary_to_trc(sspopt: str) -> str:
    """把 Bellhop 顶边界选项改成“从文件读取反射系数”的模式。

    例如：CVW -> CFW，QVW -> QFW。
    """
    text = str(sspopt)
    if len(text) < 2:
        raise ValueError("SSPOPT 至少需要两个字符")
    return text[0] + "F" + text[2:]


def stage_same_trc_for_case(trc_source: str | Path, env_path: str | Path) -> Path:
    """把区域 .trc 复制成某个 .env 算例对应的 Bellhop 文件名。"""
    src = Path(trc_source).resolve()
    env = Path(env_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if not env.is_file():
        raise FileNotFoundError(env)
    dst = env.with_suffix(".trc")
    shutil.copy2(src, dst)
    return dst


def stage_same_trc_for_many_cases(trc_source: str | Path, env_paths: Sequence[str | Path]) -> list[Path]:
    return [stage_same_trc_for_case(trc_source, env_path) for env_path in env_paths]


def _import_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime packaging.
        raise RuntimeError("读取 IS2SITMOGR4 冰数据需要 h5py") from exc
    return h5py


def _import_netcdf4():
    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime packaging.
        raise RuntimeError("读取 NetCDF 冰数据需要 netCDF4") from exc
    return netCDF4


def _import_scipy_special():
    try:
        from scipy import special  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime packaging.
        raise RuntimeError("低/中频 Mathieu 分支需要 scipy") from exc
    return special


def _find_nc(input_path: str | Path, temp_dir: str | Path | None = None) -> Path:
    path = Path(input_path).resolve()
    if path.suffix.lower() == ".nc":
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError("ice_data 必须是 .nc 或 .zip 文件")
    if not path.is_file():
        raise FileNotFoundError(path)
    work = Path(temp_dir) if temp_dir is not None else Path(tempfile.mkdtemp(prefix="ice_trc_"))
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(work)
    candidates = sorted(work.rglob("*.nc"))
    if not candidates:
        raise FileNotFoundError("ice_data ZIP 中没有找到 .nc 文件")
    return candidates[0]


def _read2d(dataset: Any, name: str) -> np.ndarray:
    data = dataset[name]
    array = np.asarray(data[...], dtype=float)
    if array.ndim == 3:
        array = array[0]
    fill = data.attrs.get("_FillValue", None)
    if fill is not None:
        array[array == float(np.asarray(fill).ravel()[0])] = np.nan
    array[array <= FILL_LIMIT] = np.nan
    array[~np.isfinite(array)] = np.nan
    return array


def _read2d_netcdf(dataset: Any, name: str) -> np.ndarray:
    data = dataset.variables[name]
    array = np.asarray(np.ma.filled(data[...], np.nan), dtype=float)
    if array.ndim == 3:
        array = array[0]
    array[array <= FILL_LIMIT] = np.nan
    array[~np.isfinite(array)] = np.nan
    return array


def _attr_text(dataset: Any, name: str) -> str:
    value = dataset.attrs.get(name, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.ravel()[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _attr_text_netcdf(variable: Any, name: str) -> str:
    value = getattr(variable, name, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _window_stack(array: np.ndarray, window_cells: int) -> np.ndarray:
    pad = window_cells // 2
    padded = np.pad(array, pad, constant_values=np.nan)
    return sliding_window_view(padded, (window_cells, window_cells))


def _local_nanpercentile(array: np.ndarray, window_cells: int, q: float, min_valid: int) -> np.ndarray:
    stack = _window_stack(array, window_cells)
    count = np.sum(np.isfinite(stack), axis=(-1, -2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            out = np.nanpercentile(stack, q, axis=(-1, -2))
    out[count < min_valid] = np.nan
    return out


def _local_nanmean(array: np.ndarray, window_cells: int, min_valid: int) -> np.ndarray:
    stack = _window_stack(array, window_cells)
    count = np.sum(np.isfinite(stack), axis=(-1, -2))
    total = np.nansum(stack, axis=(-1, -2))
    out = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    out[count < min_valid] = np.nan
    return out


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    q = np.percentile(x, [5, 25, 50, 75, 95, 99])
    mean = float(np.mean(x))
    std = float(np.std(x))
    return {
        "n": int(x.size),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 0.0 else None,
        "min": float(np.min(x)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(np.max(x)),
        "iqr": float(q[3] - q[1]),
        "relative_iqr": float((q[3] - q[1]) / q[2]) if q[2] > 0.0 else None,
    }


def derive_ice_bumden_fields(
    ice_data: str | Path,
    geometry: GeometryParams = GeometryParams(),
    processing: IceProcessingConfig = IceProcessingConfig(),
) -> dict[str, Any]:
    """读取 IS2SITMOGR4 数据并计算 BUMDEN_eff，不写出附属文件。"""
    geometry.validate()
    if processing.window_cells < 3 or processing.window_cells % 2 != 1:
        raise ValueError("window_cells 必须是不小于 3 的奇数")

    with tempfile.TemporaryDirectory(prefix="ice_trc_") as temp_dir:
        nc = _find_nc(ice_data, temp_dir)
        try:
            h5py = _import_h5py()
        except RuntimeError:
            netcdf4 = _import_netcdf4()
            with netcdf4.Dataset(nc, "r") as file:
                freeboard = _read2d_netcdf(file, "freeboard")
                snow = _read2d_netcdf(file, "snow_depth")
                thickness = _read2d_netcdf(file, "ice_thickness")
                sic = _read2d_netcdf(file, "sea_ice_conc")
                ice_type = _read2d_netcdf(file, "ice_type")
                segments = _read2d_netcdf(file, "num_segments")
                lat = np.asarray(np.ma.filled(file.variables["latitude"][...], np.nan), dtype=float)
                lon = np.asarray(np.ma.filled(file.variables["longitude"][...], np.nan), dtype=float)
                fb_name = _attr_text_netcdf(file.variables["freeboard"], "long_name").lower()
        else:
            with h5py.File(nc, "r") as file:
                freeboard = _read2d(file, "freeboard")
                snow = _read2d(file, "snow_depth")
                thickness = _read2d(file, "ice_thickness")
                sic = _read2d(file, "sea_ice_conc")
                ice_type = _read2d(file, "ice_type")
                segments = _read2d(file, "num_segments")
                lat = np.asarray(file["latitude"][...], dtype=float)
                lon = np.asarray(file["longitude"][...], dtype=float)
                fb_name = _attr_text(file["freeboard"], "long_name").lower()

    if "total freeboard" in fb_name:
        draft = thickness - freeboard + snow
    else:
        draft = thickness - freeboard

    valid = (
        np.isfinite(draft)
        & np.isfinite(sic)
        & np.isfinite(ice_type)
        & np.isfinite(segments)
        & (draft >= 0.0)
        & (sic >= processing.min_sic)
        & (segments >= processing.min_segments)
    )

    background = np.full_like(draft, np.nan)
    excess = np.full_like(draft, np.nan)
    local_excess = np.full_like(draft, np.nan)
    bumden = np.full_like(draft, np.nan)
    keel_area = math.pi * 0.5 * geometry.eta_m * geometry.xi_m

    for code in (0, 1):
        type_mask = valid & (ice_type == code)
        draft_for_type = np.where(type_mask, draft, np.nan)
        bg = _local_nanpercentile(
            draft_for_type,
            processing.window_cells,
            processing.background_percentile,
            processing.min_window_valid,
        )
        ex = np.where(type_mask & np.isfinite(bg), np.maximum(draft - bg, 0.0), np.nan)
        local_mean = _local_nanmean(ex, processing.window_cells, processing.min_window_valid)
        b = np.where(type_mask, 1000.0 * local_mean / keel_area, np.nan)
        background[type_mask] = bg[type_mask]
        excess[type_mask] = ex[type_mask]
        local_excess[type_mask] = local_mean[type_mask]
        bumden[type_mask] = b[type_mask]

    return {
        "lat": lat,
        "lon": lon,
        "ice_thickness": thickness,
        "sea_ice_conc": sic,
        "ice_type": ice_type,
        "draft": draft,
        "background": background,
        "excess": excess,
        "local_excess": local_excess,
        "bumden": bumden,
        "mask": valid,
        "stats": _stats(bumden),
    }


def haversine_km(lat0_deg: float, lon0_deg: float, lat_deg: Any, lon_deg: Any) -> np.ndarray:
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def _choose_representative(stats: Mapping[str, Any], aggregate: str) -> float:
    method = aggregate.lower()
    if method in {"median", "p50"}:
        return float(stats["median"])
    if method == "mean":
        return float(stats["mean"])
    if method == "p25":
        return float(stats["p25"])
    if method == "p75":
        return float(stats["p75"])
    raise ValueError("aggregate must be median, mean, p25, or p75")


def _select_domain_bumden(
    bumden: np.ndarray,
    valid: np.ndarray,
    distance_km: np.ndarray,
    radius_km: float,
    min_domain_cells: int,
    aggregate: str,
) -> dict[str, Any]:
    """选择计算域的代表性 BUMDEN，数据稀疏时降级而不中止任务。

    正常情况使用指定半径内的所有有效网格。若域内只有 1—2 个值，
    直接使用它们；若域内没有值，使用已加载子区域中距声源最近的
    有效值。整个子区域都无法生成 BUMDEN 时，以 0 代表无可解析
    冰脊的平整冰面。
    """
    if min_domain_cells < 1:
        raise ValueError("min_domain_cells 必须大于等于 1")
    if str(aggregate).lower() not in {"median", "p50", "mean", "p25", "p75"}:
        raise ValueError("aggregate must be median, mean, p25, or p75")

    values = np.asarray(bumden, dtype=float)
    finite_distance = np.isfinite(distance_km)
    candidates = np.asarray(valid, dtype=bool) & np.isfinite(values) & finite_distance
    requested_domain = candidates & (distance_km <= float(radius_km))
    domain_count = int(np.count_nonzero(requested_domain))
    available_count = int(np.count_nonzero(candidates))
    selected = requested_domain.copy()

    if domain_count >= min_domain_cells:
        fallback_mode = "none"
        message = "BUMDEN 网格数量满足计算要求。"
    elif domain_count > 0:
        fallback_mode = "sparse_domain"
        message = (
            f"{radius_km:g} km 范围内只有 {domain_count} 个有效 BUMDEN 网格，"
            f"少于期望的 {min_domain_cells} 个；已使用域内现有数据继续计算。"
        )
    elif available_count > 0:
        flat_candidates = np.flatnonzero(candidates.ravel())
        nearest_count = min(min_domain_cells, available_count)
        order = np.argsort(distance_km.ravel()[flat_candidates], kind="stable")[:nearest_count]
        selected = np.zeros(values.shape, dtype=bool)
        selected.ravel()[flat_candidates[order]] = True
        fallback_mode = "nearest_valid"
        nearest_distance = float(np.min(distance_km[selected]))
        message = (
            f"{radius_km:g} km 范围内没有有效 BUMDEN 网格；"
            f"已使用已加载区域中最近的 {nearest_count} 个网格"
            f"（最近距离 {nearest_distance:.2f} km）继续计算。"
        )
    else:
        fallback_mode = "flat_ice"
        message = (
            f"{radius_km:g} km 范围及已加载的邻近区域都无法生成 BUMDEN；"
            "已按 BUMDEN=0 的平整冰面继续计算。"
        )

    selected_count = int(np.count_nonzero(selected))
    stats = _stats(values[selected]) if selected_count else {}
    representative = _choose_representative(stats, aggregate) if stats else 0.0
    selected_distances = distance_km[selected]
    selection = {
        "fallback_mode": fallback_mode,
        "degraded": fallback_mode != "none",
        "message": message,
        "requested_radius_km": float(radius_km),
        "requested_min_cell_count": int(min_domain_cells),
        "domain_valid_cell_count": domain_count,
        "available_valid_cell_count": available_count,
        "selected_cell_count": selected_count,
        "selected_distance_min_km": (
            float(np.min(selected_distances)) if selected_count else None
        ),
        "selected_distance_max_km": (
            float(np.max(selected_distances)) if selected_count else None
        ),
    }
    return {
        "mask": selected,
        "BUMDEN_domain": float(representative),
        "stats": stats,
        "selection": selection,
    }


def _normalise_ice_data_source(data_source: str | None = None) -> str:
    source = str(data_source or ICE_REFLECTION_DATA_SOURCE).strip().upper().replace("-", "")
    aliases = {
        "ICESAT2": "ICESAT2",
        "IS2SITMOGR4": "ICESAT2",
        "HYCOM": "HYCOM",
        "CICE": "HYCOM",
        "HYCOMCICE": "HYCOM",
        "NEXTSIM": "NEXTSIM",
    }
    try:
        return aliases[source]
    except KeyError as exc:
        raise ValueError(
            "冰反射数据源必须是 HYCOM、NEXTSIM 或 ICESAT2；"
            f"当前值为 {data_source or ICE_REFLECTION_DATA_SOURCE!r}"
        ) from exc


def selected_ice_data_path(
    ice_data: str | Path | None,
    data_source: str | None = None,
) -> Path | None:
    """返回当前宏式入口实际使用的冰数据文件。"""
    source = _normalise_ice_data_source(data_source)
    if source == "HYCOM" and HYCOM_ICE_DATA_FILE is not None:
        return Path(HYCOM_ICE_DATA_FILE).resolve()
    if source == "NEXTSIM" and NEXTSIM_ICE_DATA_FILE is not None:
        return Path(NEXTSIM_ICE_DATA_FILE).resolve()
    if source == "ICESAT2" and ICESAT2_ICE_DATA_FILE is not None:
        return Path(ICESAT2_ICE_DATA_FILE).resolve()
    return Path(ice_data) if ice_data is not None else None


def reflection_data_kind(data_source: str | None = None) -> str | None:
    """返回门户数据路由键；ICESat-2 必须使用显式文件路径。"""
    source = _normalise_ice_data_source(data_source)
    return {"HYCOM": "ice", "NEXTSIM": "nextsim", "ICESAT2": None}[source]


def _nextsim_processing(base: HycomProcessingConfig) -> HycomProcessingConfig:
    """复用相同冰脊参数化，只替换 neXtSIM 的 NetCDF 字段名称。"""
    return replace(
        base,
        sic_variable=NEXTSIM_SIC_VARIABLE,
        sih_variable=NEXTSIM_SIH_VARIABLE,
        lat_variable=NEXTSIM_LAT_VARIABLE,
        lon_variable=NEXTSIM_LON_VARIABLE,
    )


def _open_hycom_netcdf(path: str | Path) -> Any:
    """打开 HYCOM NetCDF；Windows 中文路径不兼容时退回内存读取。"""
    nc_path = Path(path).resolve()
    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)
    netcdf4 = _import_netcdf4()
    try:
        return netcdf4.Dataset(str(nc_path), "r")
    except OSError:
        return netcdf4.Dataset("hycom_ice_in_memory.nc", "r", memory=nc_path.read_bytes())


def _hycom_subset_axes(
    latitude: np.ndarray,
    longitude: np.ndarray,
    source_lat: float,
    source_lon: float,
    radius_km: float,
    window_cells: int,
) -> tuple[slice, np.ndarray, np.ndarray, np.ndarray]:
    """选择计算域和局部统计窗口所需的小块规则网格。"""
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("HYCOM lat/lon 必须是一维规则网格坐标")
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("HYCOM lat/lon 网格点不足")

    lat_steps = np.abs(np.diff(latitude[np.isfinite(latitude)]))
    lon_sorted = np.sort(np.mod(longitude[np.isfinite(longitude)], 360.0))
    lon_steps = np.diff(lon_sorted)
    lat_steps = lat_steps[(lat_steps > 0.0) & np.isfinite(lat_steps)]
    lon_steps = lon_steps[(lon_steps > 0.0) & (lon_steps < 180.0) & np.isfinite(lon_steps)]
    if lat_steps.size == 0 or lon_steps.size == 0:
        raise ValueError("无法确定 HYCOM 网格分辨率")

    context_cells = window_cells // 2 + 1
    lat_step_deg = float(np.median(lat_steps))
    lon_step_deg = float(np.median(lon_steps))
    lat_margin_deg = radius_km / 110.574 + context_cells * lat_step_deg
    cos_lat = max(abs(math.cos(math.radians(float(source_lat)))), 1.0e-3)
    lon_margin_deg = min(180.0, radius_km / (111.320 * cos_lat) + context_cells * lon_step_deg)

    lat_indices = np.flatnonzero(
        np.isfinite(latitude) & (np.abs(latitude - float(source_lat)) <= lat_margin_deg)
    )
    source_lon_360 = float(source_lon) % 360.0
    lon_delta = (np.mod(longitude, 360.0) - source_lon_360 + 180.0) % 360.0 - 180.0
    lon_indices = np.flatnonzero(np.isfinite(longitude) & (np.abs(lon_delta) <= lon_margin_deg))
    if lat_indices.size == 0 or lon_indices.size == 0:
        raise RuntimeError("声源计算域不在 HYCOM 海冰网格范围内")

    row_slice = slice(int(lat_indices.min()), int(lat_indices.max()) + 1)
    lon_indices = lon_indices[np.argsort(lon_delta[lon_indices])]
    return row_slice, lon_indices, latitude[row_slice], longitude[lon_indices]


def _read_hycom_plane(
    dataset: Any,
    variable_name: str,
    time_index: int,
    row_slice: slice,
    lon_indices: np.ndarray,
    expected_grid_shape: tuple[int, int],
) -> np.ndarray:
    if variable_name not in dataset.variables:
        raise KeyError(f"HYCOM 文件缺少变量 {variable_name!r}")
    variable = dataset.variables[variable_name]
    if tuple(variable.shape[-2:]) != expected_grid_shape:
        raise ValueError(
            f"HYCOM 变量 {variable_name!r} 的末两维 {variable.shape[-2:]} "
            f"与 lat/lon 网格 {expected_grid_shape} 不一致"
        )
    if variable.ndim == 3:
        if time_index >= variable.shape[0]:
            raise IndexError(
                f"HYCOM time_index={time_index} 超出变量 {variable_name!r} 的时间维"
            )
        values = variable[time_index, row_slice, :]
    elif variable.ndim == 2:
        values = variable[row_slice, :]
    else:
        raise ValueError(f"HYCOM 变量 {variable_name!r} 必须是二维或三维")
    array = np.asarray(np.ma.filled(values, np.nan), dtype=float)[:, lon_indices]
    array[~np.isfinite(array)] = np.nan
    return array


def derive_hycom_ice_fields(
    ice_data: str | Path,
    source_lat: float,
    source_lon: float,
    radius_km: float,
    processing: HycomProcessingConfig = HycomProcessingConfig(),
) -> dict[str, Any]:
    """读取声源附近的 HYCOM 海冰浓度和平均冰厚网格。

    门户接口只需要沿传播路径查询冰厚，因此这里复用 HYCOM 的局部网格
    截取逻辑，避免把全球 ``sic``/``sih`` 平面长期缓存在内存中。
    """
    processing.validate()
    if radius_km <= 0.0:
        raise ValueError("radius_km 必须大于 0")

    dataset = _open_hycom_netcdf(ice_data)
    try:
        for coordinate in (processing.lat_variable, processing.lon_variable):
            if coordinate not in dataset.variables:
                raise KeyError(f"HYCOM 文件缺少坐标变量 {coordinate!r}")
        lat_all = np.asarray(
            np.ma.filled(dataset.variables[processing.lat_variable][...], np.nan), dtype=float
        )
        lon_all = np.asarray(
            np.ma.filled(dataset.variables[processing.lon_variable][...], np.nan), dtype=float
        )
        row_slice, lon_indices, lat, lon = _hycom_subset_axes(
            lat_all,
            lon_all,
            source_lat,
            source_lon,
            radius_km,
            processing.window_cells,
        )
        grid_shape = (lat_all.size, lon_all.size)
        sic = _read_hycom_plane(
            dataset,
            processing.sic_variable,
            processing.time_index,
            row_slice,
            lon_indices,
            grid_shape,
        )
        thickness = _read_hycom_plane(
            dataset,
            processing.sih_variable,
            processing.time_index,
            row_slice,
            lon_indices,
            grid_shape,
        )
    finally:
        dataset.close()

    finite_sic = sic[np.isfinite(sic)]
    if finite_sic.size and float(np.nanmax(finite_sic)) > 1.5:
        sic = sic / 100.0
    sic = np.clip(sic, 0.0, 1.0)
    observed = np.isfinite(thickness) & np.isfinite(sic) & (thickness >= 0.0)
    valid = observed & (sic >= processing.min_sic)
    # ``valid`` 仍是反射模型使用的冰覆盖掩膜，但冰厚产品也会返回门户用于绘图。
    # 冰浓度低于阈值且数值有限的海洋网格表示开阔水域，而不是数据缺失：将其显示为
    # 0 m，避免响应序列化程序将其转换成缺测值标记 9999。
    display_thickness = np.where(observed, np.where(valid, thickness, 0.0), np.nan)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return {
        "lat": lat_grid,
        "lon": lon_grid,
        "ice_thickness": display_thickness,
        "sea_ice_conc": np.where(valid, sic, np.nan),
        "mask": valid,
        "data_source": "HYCOM",
    }


def derive_hycom_bumden_domain(
    ice_data: str | Path,
    source_lat: float,
    source_lon: float,
    radius_km: float,
    geometry: GeometryParams = GeometryParams(),
    processing: HycomProcessingConfig = HycomProcessingConfig(),
    aggregate: str = "median",
    min_domain_cells: int = 3,
) -> dict[str, Any]:
    """从 HYCOM/CICE 的 sic、sih 参数化估算计算域等效 BUMDEN。

    sih 先按静力平衡近似换算为平均冰吃水；局部低分位数代表平整冰背景，
    正厚度异常代表未解析的等效冰脊体积。该结果是工程参数化量，不是冰脊实测值。
    """
    geometry.validate()
    processing.validate()
    if radius_km <= 0.0:
        raise ValueError("radius_km 必须大于 0")
    if min_domain_cells < 1:
        raise ValueError("min_domain_cells 必须大于等于 1")

    dataset = _open_hycom_netcdf(ice_data)
    try:
        for coordinate in (processing.lat_variable, processing.lon_variable):
            if coordinate not in dataset.variables:
                raise KeyError(f"HYCOM 文件缺少坐标变量 {coordinate!r}")
        lat_all = np.asarray(
            np.ma.filled(dataset.variables[processing.lat_variable][...], np.nan), dtype=float
        )
        lon_all = np.asarray(
            np.ma.filled(dataset.variables[processing.lon_variable][...], np.nan), dtype=float
        )
        row_slice, lon_indices, lat, lon = _hycom_subset_axes(
            lat_all,
            lon_all,
            source_lat,
            source_lon,
            radius_km,
            processing.window_cells,
        )
        grid_shape = (lat_all.size, lon_all.size)
        sic = _read_hycom_plane(
            dataset,
            processing.sic_variable,
            processing.time_index,
            row_slice,
            lon_indices,
            grid_shape,
        )
        thickness = _read_hycom_plane(
            dataset,
            processing.sih_variable,
            processing.time_index,
            row_slice,
            lon_indices,
            grid_shape,
        )

        finite_sic = sic[np.isfinite(sic)]
        if finite_sic.size and float(np.nanmax(finite_sic)) > 1.5:
            sic = sic / 100.0
        sic = np.clip(sic, 0.0, 1.0)

        time_info: dict[str, Any] = {"time_index": int(processing.time_index)}
        time_variable_name = processing.time_variable
        if time_variable_name not in dataset.variables:
            time_variable_name = next(
                (
                    name
                    for name, variable in dataset.variables.items()
                    if str(getattr(variable, "standard_name", "")).strip().lower() == "time"
                    or str(getattr(variable, "_CoordinateAxisType", "")).strip().lower() == "time"
                ),
                "",
            )
        if time_variable_name:
            time_variable = dataset.variables[time_variable_name]
            time_info["variable"] = str(time_variable_name)
            if processing.time_index < time_variable.shape[0]:
                time_value = float(time_variable[processing.time_index])
                time_info["value"] = time_value
                time_info["units"] = str(getattr(time_variable, "units", ""))
                try:
                    netcdf4 = _import_netcdf4()
                    calendar = str(getattr(time_variable, "calendar", "standard"))
                    decoded = netcdf4.num2date(time_value, time_info["units"], calendar=calendar)
                    time_info["valid_time"] = str(decoded)
                except Exception:
                    pass
        source_info = {
            name: str(getattr(dataset, name))
            for name in ("source", "generating_model", "grid_name", "time_origin")
            if hasattr(dataset, name)
        }
    finally:
        dataset.close()

    valid = (
        np.isfinite(thickness)
        & np.isfinite(sic)
        & (thickness >= 0.0)
        & (sic >= processing.min_sic)
    )
    draft_ratio = processing.ice_density_kg_m3 / processing.water_density_kg_m3
    draft = np.where(valid, thickness * draft_ratio, np.nan)
    background = _local_nanpercentile(
        draft,
        processing.window_cells,
        processing.background_percentile,
        processing.min_window_valid,
    )
    excess = np.where(valid & np.isfinite(background), np.maximum(draft - background, 0.0), np.nan)
    local_excess = _local_nanmean(
        excess,
        processing.window_cells,
        processing.min_window_valid,
    )
    keel_area_m2 = math.pi * 0.5 * geometry.eta_m * geometry.xi_m
    bumden = 1000.0 * local_excess / keel_area_m2
    if processing.weight_by_sic:
        bumden = bumden * sic
    bumden = bumden * processing.bumden_scale
    bumden = np.where(valid & np.isfinite(bumden), np.maximum(bumden, 0.0), np.nan)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    distance = haversine_km(source_lat, source_lon, lat_grid, lon_grid)
    domain_ice = valid & np.isfinite(distance) & (distance <= radius_km)
    selected = _select_domain_bumden(
        bumden,
        valid,
        distance,
        radius_km,
        min_domain_cells,
        aggregate,
    )
    domain = selected["mask"]
    domain_stats = selected["stats"]
    ice_stats = {
        "thickness_m": _stats(thickness[domain_ice]),
        "sic_fraction": _stats(sic[domain_ice]),
        "estimated_draft_m": _stats(draft[domain_ice]),
        "local_excess_draft_m": _stats(local_excess[domain]),
        "bumden_ridges_per_km": domain_stats,
    }
    return {
        "BUMDEN_domain": selected["BUMDEN_domain"],
        "domain_stats": domain_stats,
        "ice_stats": ice_stats,
        "selection": selected["selection"],
        "data_source": "HYCOM",
        "parameterization": {
            "description": "HYCOM sih spatial anomaly converted to equivalent ridge density",
            "measured_ridge_density": False,
            "draft_ratio": float(draft_ratio),
            "keel_area_m2": float(keel_area_m2),
            "weight_by_sic": bool(processing.weight_by_sic),
            "bumden_scale": float(processing.bumden_scale),
            "window_cells": int(processing.window_cells),
            "background_percentile": float(processing.background_percentile),
            "min_sic": float(processing.min_sic),
        },
        "time": time_info,
        "source_metadata": source_info,
    }


def derive_domain_bumden(
    fields: Mapping[str, Any],
    source_lat: float,
    source_lon: float,
    config: DomainConfig,
) -> dict[str, Any]:
    """为声源中心圆形区域计算一个代表性 BUMDEN_domain。"""
    if config.radius_km <= 0.0:
        raise ValueError("radius_km 必须大于 0")
    lat = np.asarray(fields["lat"], dtype=float)
    lon = np.asarray(fields["lon"], dtype=float)
    bumden = np.asarray(fields["bumden"], dtype=float)
    valid = np.asarray(fields["mask"], dtype=bool) & np.isfinite(bumden)
    distance = haversine_km(source_lat, source_lon, lat, lon)
    selected = _select_domain_bumden(
        bumden,
        valid,
        distance,
        config.radius_km,
        config.min_domain_cells,
        config.aggregate,
    )
    return {
        "mask": selected["mask"],
        "distance_km": distance,
        "BUMDEN_domain": selected["BUMDEN_domain"],
        "stats": selected["stats"],
        "selection": selected["selection"],
    }


def _radial_even_out(m: int, q: float, xi: float) -> complex:
    special = _import_scipy_special()
    return special.mathieu_modcem1(m, q, xi)[0] + 1j * special.mathieu_modcem2(m, q, xi)[0]


def _radial_odd_out(m: int, q: float, xi: float) -> complex:
    special = _import_scipy_special()
    return special.mathieu_modsem1(m, q, xi)[0] + 1j * special.mathieu_modsem2(m, q, xi)[0]


def _ang_even(m: int, q: float, eta_rad: np.ndarray) -> np.ndarray:
    special = _import_scipy_special()
    return special.mathieu_cem(m, q, np.rad2deg(eta_rad))[0]


def _ang_odd(m: int, q: float, eta_rad: np.ndarray) -> np.ndarray:
    special = _import_scipy_special()
    return special.mathieu_sem(m, q, np.rad2deg(eta_rad))[0]


def _elliptic_coordinates(x: np.ndarray, y: np.ndarray, focal_c: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.arccosh((np.asarray(x) + 1j * np.asarray(y)) / focal_c)
    return np.real(w), np.imag(w)


def _choose_nmax(ka: float) -> int:
    return max(12, int(math.ceil(ka + 12.0)))


def _choose_nquad(nmax: int) -> int:
    return max(1024, 16 * (2 * nmax + 1))


def _build_mathieu_basis(q: float, xi0: float, eta_values: np.ndarray, nmax: int):
    basis: list[tuple[str, int, np.ndarray]] = []
    h0: list[complex] = []
    for m in range(0, nmax + 1):
        s = _ang_even(m, q, eta_values)
        h = _radial_even_out(m, q, xi0)
        if np.all(np.isfinite(s)) and np.isfinite(h) and abs(h) > 1e-300:
            basis.append(("even", m, s))
            h0.append(h)
    for m in range(1, nmax + 1):
        s = _ang_odd(m, q, eta_values)
        h = _radial_odd_out(m, q, xi0)
        if np.all(np.isfinite(s)) and np.isfinite(h) and abs(h) > 1e-300:
            basis.append(("odd", m, s))
            h0.append(h)
    if not basis:
        raise RuntimeError("当前频率下没有可用的 Mathieu 模态")
    return basis, np.asarray(h0, dtype=np.complex128)


def _incident_coefficients(
    k: float,
    a: float,
    b: float,
    eta_values: np.ndarray,
    basis_matrix: np.ndarray,
    norms: np.ndarray,
    h0: np.ndarray,
    alpha_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xb = a * np.cos(eta_values)
    yb = b * np.sin(eta_values)
    alpha = np.asarray(alpha_rad, dtype=float)
    u = np.exp(1j * k * (np.cos(alpha)[:, None] * xb[None, :] + np.sin(alpha)[:, None] * yb[None, :]))
    c = (u @ basis_matrix.T) / norms[None, :]
    amplitudes = -c / h0[None, :]
    residual = u + (amplitudes * h0[None, :]) @ basis_matrix
    rms = np.sqrt(np.mean(np.abs(residual) ** 2, axis=1))
    max_abs = np.max(np.abs(residual), axis=1)
    return amplitudes, rms, max_abs


def _far_field_g(
    k: float,
    focal_c: float,
    q: float,
    basis: Sequence[tuple[str, int, np.ndarray]],
    amplitudes: np.ndarray,
    phi: np.ndarray,
    r_far: float,
) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    x = r_far * np.cos(phi)
    y = r_far * np.sin(phi)
    xi_obs, eta_obs = _elliptic_coordinates(x, y, focal_c)
    u = np.zeros(phi.size, dtype=np.complex128)
    for j, (kind, m, _) in enumerate(basis):
        if kind == "even":
            s_obs = _ang_even(m, q, eta_obs)
            h_obs = _radial_even_out(m, q, xi_obs)
        else:
            s_obs = _ang_odd(m, q, eta_obs)
            h_obs = _radial_odd_out(m, q, xi_obs)
        u += amplitudes[:, j] * s_obs * h_obs
    factor = np.exp(1j * k * r_far) * np.sqrt(2.0 / (1j * math.pi * k * r_far))
    return u / factor


def semiellipse_boss_scattering(
    frequency_hz: float,
    grazing_angle_deg: Sequence[float],
    geometry: GeometryParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry.validate()
    theta_deg = np.asarray(grazing_angle_deg, dtype=float)
    if theta_deg.ndim != 1 or np.any(theta_deg <= 0.0) or np.any(theta_deg >= 90.0):
        raise ValueError("掠射角必须是一维数组，并满足 0 < angle < 90 deg")
    frequency = float(frequency_hz)
    if frequency <= 0.0:
        raise ValueError("frequency_hz 必须大于 0")

    a = geometry.eta_m
    b = geometry.xi_m
    c0 = geometry.sound_speed_m_s
    k = 2.0 * math.pi * frequency / c0
    ka = k * a
    focal_c = math.sqrt(a * a - b * b)
    q = (k * focal_c / 2.0) ** 2
    xi0 = math.acosh(a / focal_c)
    nmax = _choose_nmax(ka)
    nquad = _choose_nquad(nmax)
    eta_values = np.linspace(0.0, 2.0 * math.pi, nquad, endpoint=False)
    basis, h0 = _build_mathieu_basis(q, xi0, eta_values, nmax)
    basis_matrix = np.asarray([item[2] for item in basis], dtype=float)
    norms = np.sum(basis_matrix * basis_matrix, axis=1)
    theta = np.deg2rad(theta_deg)
    a_down, rms_down, max_down = _incident_coefficients(k, a, b, eta_values, basis_matrix, norms, h0, -theta)
    a_img, rms_img, max_img = _incident_coefficients(k, a, b, eta_values, basis_matrix, norms, h0, theta)
    r_far = max(5000.0, 100.0 / k, 500.0 * a)
    g_down = _far_field_g(k, focal_c, q, basis, a_down, theta, r_far)
    g_img = _far_field_g(k, focal_c, q, basis, a_img, theta, r_far)
    f_boss = g_img - g_down
    residual_rms = np.maximum(rms_down, rms_img)
    residual_max = np.maximum(max_down, max_img)
    expected = 2 * nmax + 1
    valid = bool(np.all(np.isfinite(f_boss)) and np.max(residual_rms) <= geometry.residual_tol)
    return f_boss, {
        "frequency_hz": frequency,
        "k_1_m": float(k),
        "ka": float(ka),
        "kb": float(k * b),
        "q_mathieu": float(q),
        "focal_c_m": float(focal_c),
        "xi0": float(xi0),
        "nmax": int(nmax),
        "expected_modes": int(expected),
        "used_modes": int(len(basis)),
        "used_mode_fraction": float(len(basis) / expected),
        "nquad": int(nquad),
        "r_far_m": float(r_far),
        "boundary_residual_rms_by_angle": residual_rms,
        "boundary_residual_max_by_angle": residual_max,
        "boundary_residual_rms_max": float(np.max(residual_rms)),
        "boundary_residual_abs_max": float(np.max(residual_max)),
        "solver_valid": valid,
    }


def mfs_semiellipse_boss_scattering(
    frequency_hz: float,
    grazing_angle_deg: Sequence[float],
    geometry: GeometryParams,
    *,
    source_count: int = 192,
    source_scale: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the pressure-release ellipse with interior fundamental sources.

    This is an independent fallback for the high-q region where SciPy's angular
    Mathieu functions can become numerically rank deficient.  It enforces the
    same full-ellipse Dirichlet boundary condition and uses the same image
    construction as :func:`semiellipse_boss_scattering`.
    """
    geometry.validate()
    theta_deg = np.asarray(grazing_angle_deg, dtype=float)
    if theta_deg.ndim != 1 or np.any(theta_deg <= 0.0) or np.any(theta_deg >= 90.0):
        raise ValueError("掠射角必须是一维数组，并满足 0 < angle < 90 deg")
    frequency = float(frequency_hz)
    if frequency <= 0.0:
        raise ValueError("frequency_hz 必须大于 0")
    if int(source_count) < 32:
        raise ValueError("MFS source_count 必须大于等于 32")
    if not 0.0 < float(source_scale) < 1.0:
        raise ValueError("MFS source_scale 必须严格位于 0 到 1 之间")

    special = _import_scipy_special()
    a = float(geometry.eta_m)
    b = float(geometry.xi_m)
    c0 = float(geometry.sound_speed_m_s)
    k = 2.0 * math.pi * frequency / c0
    count = int(source_count)
    ncollocation = 3 * count
    tc = (np.arange(ncollocation, dtype=float) + 0.31) * (2.0 * math.pi / ncollocation)
    ts = (np.arange(count, dtype=float) + 0.07) * (2.0 * math.pi / count)
    xc = a * np.cos(tc)
    yc = b * np.sin(tc)
    xs = float(source_scale) * a * np.cos(ts)
    ys = float(source_scale) * b * np.sin(ts)
    distance = np.hypot(xc[:, None] - xs[None, :], yc[:, None] - ys[None, :])
    raw_matrix = special.hankel1(0, k * distance)
    column_scale = np.linalg.norm(raw_matrix, axis=0)
    matrix = raw_matrix / column_scale[None, :]

    theta = np.deg2rad(theta_deg)
    incident_angles = np.concatenate((-theta, theta))
    incident = np.exp(
        1j * k * (
            np.cos(incident_angles)[:, None] * xc[None, :]
            + np.sin(incident_angles)[:, None] * yc[None, :]
        )
    )
    scaled_coefficients, _, rank, singular = np.linalg.lstsq(
        matrix, -incident.T, rcond=1.0e-12
    )
    coefficients = scaled_coefficients / column_scale[:, None]

    ntest = 4 * count
    tt = (np.arange(ntest, dtype=float) + 0.73) * (2.0 * math.pi / ntest)
    xt = a * np.cos(tt)
    yt = b * np.sin(tt)
    distance_test = np.hypot(xt[:, None] - xs[None, :], yt[:, None] - ys[None, :])
    test_matrix = special.hankel1(0, k * distance_test)
    incident_test = np.exp(
        1j * k * (
            np.cos(incident_angles)[:, None] * xt[None, :]
            + np.sin(incident_angles)[:, None] * yt[None, :]
        )
    )
    residual = incident_test + (test_matrix @ coefficients).T
    residual_rms = np.sqrt(np.mean(np.abs(residual) ** 2, axis=1))
    residual_max = np.max(np.abs(residual), axis=1)

    observation = np.concatenate((theta, theta))
    far_basis = np.exp(
        -1j * k * (
            np.cos(observation)[:, None] * xs[None, :]
            + np.sin(observation)[:, None] * ys[None, :]
        )
    )
    far_field = np.einsum("jm,mj->j", far_basis, coefficients)
    nangle = theta.size
    f_boss = far_field[nangle:] - far_field[:nangle]
    residual_limit = max(float(geometry.residual_tol), 1.0e-5)
    valid = bool(
        np.all(np.isfinite(f_boss))
        and np.all(np.isfinite(residual_rms))
        and float(np.max(residual_rms)) <= residual_limit
    )
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")
    return f_boss, {
        "frequency_hz": frequency,
        "k_1_m": float(k),
        "ka": float(k * a),
        "kb": float(k * b),
        "method": "BT_MFS_DIRICHLET_IMAGE",
        "source_count": count,
        "source_scale": float(source_scale),
        "collocation_count": int(ncollocation),
        "test_count": int(ntest),
        "effective_rank": int(rank),
        "condition_number_scaled": condition,
        "boundary_residual_rms_by_angle": residual_rms,
        "boundary_residual_max_by_angle": residual_max,
        "boundary_residual_rms_max": float(np.max(residual_rms)),
        "boundary_residual_abs_max": float(np.max(residual_max)),
        "residual_limit": residual_limit,
        "solver_valid": valid,
    }


def _ramanujan_perimeter(a: float, b: float) -> float:
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _boundary_grid(a: float, b: float, wavelength: float, ppw: float, nmin: int):
    perimeter = _ramanujan_perimeter(a, b)
    n = max(int(nmin), int(math.ceil(perimeter / wavelength * ppw)))
    t = (np.arange(n, dtype=float) + 0.5) * (2.0 * math.pi / n)
    dt = 2.0 * math.pi / n
    x = a * np.cos(t)
    y = b * np.sin(t)
    dxdt = -a * np.sin(t)
    dydt = b * np.cos(t)
    dsdt = np.sqrt(dxdt * dxdt + dydt * dydt)
    nx = np.cos(t) / a
    ny = np.sin(t) / b
    norm = np.sqrt(nx * nx + ny * ny)
    return x, y, nx / norm, ny / norm, dsdt, dt, n, perimeter


def _po_ellipse_scattering(
    frequency_hz: float,
    alpha_rad: np.ndarray,
    observation_rad: np.ndarray,
    a: float,
    b: float,
    c0: float,
    ppw: float,
    nmin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    frequency = float(frequency_hz)
    k = 2.0 * math.pi * frequency / c0
    wavelength = c0 / frequency
    x, y, nx, ny, dsdt, dt, npts, perimeter = _boundary_grid(a, b, wavelength, ppw, nmin)
    alpha = np.asarray(alpha_rad, dtype=float)
    obs = np.asarray(observation_rad, dtype=float)
    if alpha.shape != obs.shape or alpha.ndim != 1:
        raise ValueError("alpha_rad 和 observation_rad 必须是一一对应的一维数组")

    g = np.empty(alpha.size, dtype=np.complex128)
    for j, (al, ph) in enumerate(zip(alpha, obs)):
        dix = math.cos(al)
        diy = math.sin(al)
        dsx = math.cos(ph)
        dsy = math.sin(ph)
        ndot = nx * dix + ny * diy
        illuminated = ndot < 0.0
        phase = np.exp(1j * k * ((dix - dsx) * x + (diy - dsy) * y))
        g[j] = 0.5 * k * np.sum(ndot[illuminated] * phase[illuminated] * dsdt[illuminated]) * dt
    return g, {
        "boundary_points": int(npts),
        "ellipse_perimeter_m": float(perimeter),
        "points_per_wavelength_requested": float(ppw),
        "wavelength_m": float(wavelength),
    }


def high_frequency_semiellipse_boss(
    frequency_hz: float,
    grazing_angle_deg: Sequence[float],
    geometry: GeometryParams,
    config: HFConfig = HFConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    config.validate()
    geometry.validate()
    theta_deg = np.asarray(grazing_angle_deg, dtype=float)
    if theta_deg.ndim != 1 or np.any(theta_deg <= 0.0) or np.any(theta_deg >= 90.0):
        raise ValueError("掠射角必须是一维数组，并满足 0 < angle < 90 deg")
    frequency = float(frequency_hz)
    if frequency <= 0.0:
        raise ValueError("frequency_hz 必须大于 0")

    a = float(geometry.eta_m)
    b = float(geometry.xi_m)
    c0 = float(geometry.sound_speed_m_s)
    k = 2.0 * math.pi * frequency / c0
    theta = np.deg2rad(theta_deg)

    g_down, fine_info = _po_ellipse_scattering(
        frequency,
        -theta,
        theta,
        a,
        b,
        c0,
        config.points_per_wavelength,
        config.min_boundary_points,
    )
    g_img, _ = _po_ellipse_scattering(
        frequency,
        theta,
        theta,
        a,
        b,
        c0,
        config.points_per_wavelength,
        config.min_boundary_points,
    )
    f_fine = g_img - g_down

    g_down_c, coarse_info = _po_ellipse_scattering(
        frequency,
        -theta,
        theta,
        a,
        b,
        c0,
        config.convergence_points_per_wavelength,
        config.min_boundary_points,
    )
    g_img_c, _ = _po_ellipse_scattering(
        frequency,
        theta,
        theta,
        a,
        b,
        c0,
        config.convergence_points_per_wavelength,
        config.min_boundary_points,
    )
    f_coarse = g_img_c - g_down_c

    denom_l2 = max(float(np.linalg.norm(f_fine)), 1e-15)
    conv_l2 = float(np.linalg.norm(f_fine - f_coarse) / denom_l2)
    denom_max = max(float(np.max(np.abs(f_fine))), 1e-15)
    conv_max = float(np.max(np.abs(f_fine - f_coarse)) / denom_max)
    ka = k * a
    kb = k * b
    kmin = min(ka, kb)
    return f_fine, {
        "frequency_hz": frequency,
        "k_1_m": float(k),
        "ka": float(ka),
        "kb": float(kb),
        "kmin": float(kmin),
        "wavelength_m": float(c0 / frequency),
        "hf_recommended": bool(kmin >= config.recommended_kmin),
        "hf_recommended_kmin_threshold": float(config.recommended_kmin),
        "boundary_points_fine": fine_info["boundary_points"],
        "boundary_points_coarse": coarse_info["boundary_points"],
        "quadrature_relative_l2_change": conv_l2,
        "quadrature_relative_max_change": conv_max,
        "solver_valid": bool(np.all(np.isfinite(f_fine)) and conv_l2 < 5e-3 and conv_max < 1e-2),
        "method": "BT_HF_NUMERICAL_KIRCHHOFF_IMAGE",
    }


def _smoothstep01(x: float) -> float:
    value = float(np.clip(x, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def hybrid_semiellipse_boss(
    frequency_hz: float,
    grazing_angle_deg: Sequence[float],
    geometry: GeometryParams,
    config: HybridConfig = HybridConfig(),
    force_model: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    config.validate()
    force = force_model.lower()
    k = 2.0 * math.pi * float(frequency_hz) / geometry.sound_speed_m_s
    dims = {
        "frequency_hz": float(frequency_hz),
        "k_1_m": float(k),
        "ka": float(k * geometry.eta_m),
        "kb": float(k * geometry.xi_m),
        "kmin": float(k * min(geometry.eta_m, geometry.xi_m)),
    }
    kb = dims["kb"]

    if force == "mathieu":
        boss, diag = semiellipse_boss_scattering(frequency_hz, grazing_angle_deg, geometry)
        diag = dict(diag)
        diag.update({"selected_model": "MATHIEU_FORCED", "hybrid_weight_hf": 0.0})
        return boss, diag

    if force in {"hf", "high_frequency", "kirchhoff"}:
        boss, diag = high_frequency_semiellipse_boss(frequency_hz, grazing_angle_deg, geometry, config.hf_config)
        diag = dict(diag)
        diag.update({"selected_model": "BT_HF_KIRCHHOFF_FORCED", "hybrid_weight_hf": 1.0})
        return boss, diag

    if force != "auto":
        raise ValueError("force_model 必须是 auto、mathieu 或 high_frequency")

    if kb <= config.blend_kb_start:
        try:
            boss, diag = semiellipse_boss_scattering(frequency_hz, grazing_angle_deg, geometry)
            if diag["solver_valid"]:
                diag = dict(diag)
                diag.update({"selected_model": "MATHIEU", "hybrid_weight_hf": 0.0})
                return boss, diag
        except Exception as exc:
            diag = {"solver_valid": False, "mathieu_exception": repr(exc)}
        try:
            boss_mfs, diag_mfs = mfs_semiellipse_boss_scattering(
                frequency_hz, grazing_angle_deg, geometry
            )
            if diag_mfs["solver_valid"]:
                diag_mfs = dict(diag_mfs)
                diag_mfs.update(
                    {
                        "selected_model": "MFS_FALLBACK_AFTER_MATHIEU_FAILURE",
                        "hybrid_weight_hf": 0.0,
                        "mathieu_diagnostics": diag,
                    }
                )
                return boss_mfs, diag_mfs
        except Exception as exc:
            diag_mfs = {"solver_valid": False, "mfs_exception": repr(exc)}
        boss_hf, diag_hf = high_frequency_semiellipse_boss(
            frequency_hz, grazing_angle_deg, geometry, config.hf_config
        )
        diag_hf = dict(diag_hf)
        diag_hf.update(
            {
                "selected_model": "HF_FALLBACK_AFTER_MATHIEU_FAILURE",
                "hybrid_weight_hf": 1.0,
                "mathieu_diagnostics": diag,
                "mfs_diagnostics": diag_mfs,
            }
        )
        return boss_hf, diag_hf

    if kb >= config.blend_kb_end:
        boss, diag = high_frequency_semiellipse_boss(frequency_hz, grazing_angle_deg, geometry, config.hf_config)
        diag = dict(diag)
        diag.update({"selected_model": "BT_HF_KIRCHHOFF", "hybrid_weight_hf": 1.0})
        return boss, diag

    weight = _smoothstep01((kb - config.blend_kb_start) / (config.blend_kb_end - config.blend_kb_start))
    boss_hf, diag_hf = high_frequency_semiellipse_boss(frequency_hz, grazing_angle_deg, geometry, config.hf_config)
    try:
        boss_mathieu, diag_mathieu = semiellipse_boss_scattering(frequency_hz, grazing_angle_deg, geometry)
        mathieu_valid = bool(diag_mathieu["solver_valid"])
    except Exception as exc:
        boss_mathieu = None
        diag_mathieu = {"solver_valid": False, "mathieu_exception": repr(exc)}
        mathieu_valid = False

    if mathieu_valid:
        boss = (1.0 - weight) * boss_mathieu + weight * boss_hf
        relative_l2 = float(np.linalg.norm(boss_hf - boss_mathieu) / max(float(np.linalg.norm(boss_mathieu)), 1e-15))
        return boss, {
            **dims,
            "selected_model": "MATHIEU_HF_COMPLEX_BLEND",
            "hybrid_weight_hf": weight,
            "solver_valid": bool(diag_hf["solver_valid"] and diag_mathieu["solver_valid"]),
            "mathieu_boundary_residual_rms_max": float(diag_mathieu["boundary_residual_rms_max"]),
            "hf_quadrature_relative_l2_change": float(diag_hf["quadrature_relative_l2_change"]),
            "hf_vs_mathieu_relative_complex_l2": relative_l2,
            "mathieu_diagnostics": diag_mathieu,
            "high_frequency_diagnostics": diag_hf,
        }

    try:
        boss_mfs, diag_mfs = mfs_semiellipse_boss_scattering(
            frequency_hz, grazing_angle_deg, geometry
        )
        if diag_mfs["solver_valid"]:
            diag_mfs = dict(diag_mfs)
            diag_mfs.update(
                {
                    "selected_model": "MFS_FALLBACK_AFTER_MATHIEU_FAILURE_IN_OVERLAP",
                    "hybrid_weight_hf": 0.0,
                    "mathieu_diagnostics": diag_mathieu,
                    "high_frequency_diagnostics": diag_hf,
                    "hf_vs_mfs_relative_complex_l2": float(
                        np.linalg.norm(boss_hf - boss_mfs)
                        / max(float(np.linalg.norm(boss_mfs)), 1e-15)
                    ),
                }
            )
            return boss_mfs, diag_mfs
    except Exception as exc:
        diag_mfs = {"solver_valid": False, "mfs_exception": repr(exc)}

    diag_hf = dict(diag_hf)
    diag_hf.update(
        {
            "selected_model": "BT_HF_KIRCHHOFF_MATHIEU_AND_MFS_FAILED_IN_OVERLAP",
            "hybrid_weight_hf": 1.0,
            "mathieu_diagnostics": diag_mathieu,
            "mfs_diagnostics": diag_mfs,
        }
    )
    return boss_hf, diag_hf


def reflection_from_boss_density(
    boss_scattering: Sequence[complex],
    diagnostics: Mapping[str, Any],
    grazing_angle_deg: Sequence[float],
    bumden_per_km: float,
    enforce_passive: bool = True,
    passive_tol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    theta = np.deg2rad(np.asarray(grazing_angle_deg, dtype=float))
    density_per_m = float(bumden_per_km) / 1000.0
    k = float(diagnostics["k_1_m"])
    z = density_per_m * np.asarray(boss_scattering, dtype=np.complex128) / (k * np.sin(theta))
    raw = -(1.0 + z) / (1.0 - z)
    raw_mag = np.abs(raw)
    clipped = raw_mag > (1.0 + passive_tol)
    if enforce_passive:
        scale = np.where(raw_mag > 1.0, raw_mag, 1.0)
        reflection = raw / scale
    else:
        reflection = raw
    return reflection, {
        "bumden_per_km": float(bumden_per_km),
        "density_per_m": density_per_m,
        "Z": z,
        "R_raw": raw,
        "passivity_clipped_mask": clipped,
        "passivity_clipped_count": int(np.count_nonzero(clipped)),
        "passivity_clipped_fraction": float(np.mean(clipped)),
    }


def calculate_ice_reflection(
    frequency_hz: float,
    grazing_angle_deg: Sequence[float],
    bumden_domain_ridges_per_km: float,
    *,
    geometry: GeometryParams = GeometryParams(),
    hybrid_config: HybridConfig = HybridConfig(),
    force_model: str = "auto",
    enforce_passive: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """计算与 Bellhop ``.trc`` 完全一致的冰面复反射系数。

    该入口供不需要落盘 ``.trc`` 的同步模型复用。返回值保留复数幅值和相位，
    因而可用于相干传播计算，而不仅是反射损失估算。
    """
    theta = np.asarray(grazing_angle_deg, dtype=float)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("grazing_angle_deg 必须是一维数组且至少包含两个角度")
    if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0) or np.any(theta >= 90.0):
        raise ValueError("冰面复反射系数的掠射角必须严格位于 0 到 90 度之间")
    bumden = float(bumden_domain_ridges_per_km)
    if not math.isfinite(bumden) or bumden < 0.0:
        raise ValueError("bumden_domain_ridges_per_km 必须是大于等于 0 的有限数")

    boss, model_diag = hybrid_semiellipse_boss(
        frequency_hz,
        theta,
        geometry,
        hybrid_config,
        force_model,
    )
    reflection, closure = reflection_from_boss_density(
        boss,
        model_diag,
        theta,
        bumden,
        enforce_passive=enforce_passive,
    )
    if not bool(model_diag.get("solver_valid", True)):
        raise RuntimeError(f"{float(frequency_hz):g} Hz 的冰下反射求解器校验失败。")
    return reflection, {
        "bumden_domain_ridges_per_km": bumden,
        "model": model_diag,
        "closure": {
            "passivity_clipped_count": closure["passivity_clipped_count"],
            "passivity_clipped_fraction": closure["passivity_clipped_fraction"],
        },
    }


def resolve_bumden_domain(
    ice_data: str | Path,
    source_lat: float,
    source_lon: float,
    radius_km: float,
    geometry: GeometryParams = GeometryParams(),
    processing: IceProcessingConfig = IceProcessingConfig(),
    aggregate: str = "median",
    min_domain_cells: int = 3,
    data_source: str | None = None,
    hycom_processing: HycomProcessingConfig = HycomProcessingConfig(),
) -> dict[str, Any]:
    source = _normalise_ice_data_source(data_source)
    selected_ice_data = selected_ice_data_path(ice_data, source)
    if selected_ice_data is None:
        raise ValueError(f"{source} 冰反射入口没有配置冰数据文件")
    if source in {"HYCOM", "NEXTSIM"}:
        result = derive_hycom_bumden_domain(
            selected_ice_data,
            source_lat,
            source_lon,
            radius_km,
            geometry,
            hycom_processing if source == "HYCOM" else _nextsim_processing(hycom_processing),
            aggregate,
            min_domain_cells,
        )
        if source == "NEXTSIM":
            result["data_source"] = source
            result["parameterization"]["description"] = (
                "neXtSIM sithick spatial anomaly converted to equivalent ridge density"
            )
        return result

    fields = derive_ice_bumden_fields(selected_ice_data, geometry, processing)
    domain = derive_domain_bumden(
        fields,
        source_lat,
        source_lon,
        DomainConfig(radius_km=radius_km, aggregate=aggregate, min_domain_cells=min_domain_cells),
    )
    return {
        "BUMDEN_domain": float(domain["BUMDEN_domain"]),
        "domain_stats": domain["stats"],
        "ice_stats": fields["stats"],
        "selection": domain["selection"],
        "data_source": "ICESAT2",
        "time": {"valid_month": read_icesat2_month(selected_ice_data)},
    }


def read_icesat2_month(ice_data: str | Path) -> str:
    """从文件内部 time 读取 IS2SITMOGR4 月份，不信任文件名中的时次。"""
    with tempfile.TemporaryDirectory(prefix="ice_trc_time_") as temp_dir:
        nc_path = _find_nc(ice_data, temp_dir)
        dataset = _open_hycom_netcdf(nc_path)
        try:
            if "time" not in dataset.variables:
                raise ValueError("ICESat-2 文件缺少 time 变量，无法核对月份")
            variable = dataset.variables["time"]
            if not variable.size or not getattr(variable, "units", None):
                raise ValueError("ICESat-2 文件缺少可解析的 time/units")
            date = _import_netcdf4().num2date(
                float(variable[0]),
                variable.units,
                calendar=str(getattr(variable, "calendar", "standard")),
            )
            return f"{date.year:04d}-{date.month:02d}"
        finally:
            dataset.close()


def validate_reflection_data_time(
    bumden_info: Mapping[str, Any], requested_time: str | None,
) -> None:
    """阻止静态冰文件被用于不匹配的请求时次。"""
    if not requested_time:
        return
    source = str(bumden_info.get("data_source") or "")
    time_info = bumden_info.get("time") or {}
    def hour_key(value: str) -> str:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H")

    request_key = hour_key(str(requested_time))
    if source == "ICESAT2":
        file_month = str(time_info.get("valid_month") or "")
        if not file_month or request_key[:7] != file_month:
            raise ValueError(
                f"ICESat-2 月度海冰数据为 {file_month or '未知月份'}，"
                f"与请求时间 {requested_time} 不匹配"
            )
    elif source in {"HYCOM", "NEXTSIM"}:
        valid_time = str(time_info.get("valid_time") or "")
        if not valid_time:
            raise ValueError(f"{source} 海冰文件缺少可核对的有效时次")
        if hour_key(valid_time) != request_key:
            raise ValueError(
                f"{source} 海冰时次 {valid_time} 与请求时间 {requested_time} 不匹配"
            )


def generate_ice_reflection_trc(
    output_path: str | Path,
    frequency_hz: float,
    *,
    bumden_domain_ridges_per_km: float | None = None,
    ice_data: str | Path | None = None,
    source_lat: float | None = None,
    source_lon: float | None = None,
    domain_radius_km: float | None = None,
    geometry: GeometryParams = GeometryParams(),
    processing: IceProcessingConfig = IceProcessingConfig(),
    aggregate: str = "median",
    min_domain_cells: int = 3,
    data_source: str | None = None,
    hycom_processing: HycomProcessingConfig = HycomProcessingConfig(),
    angles_deg: Sequence[float] | None = None,
    hybrid_config: HybridConfig = HybridConfig(),
    force_model: str = "auto",
    enforce_passive: bool = True,
    full_grazing_angle_domain: bool = False,
) -> TrcGenerationResult:
    """为一个冰覆盖计算区域生成一个顶边界 .trc 文件。"""
    theta = angle_grid() if angles_deg is None else np.asarray(angles_deg, dtype=float)
    if bumden_domain_ridges_per_km is None:
        selected_ice_data = selected_ice_data_path(ice_data, data_source)
        if selected_ice_data is None or source_lat is None or source_lon is None or domain_radius_km is None:
            raise ValueError(
                "请提供 bumden_domain_ridges_per_km；或者同时提供 ice_data、"
                "source_lat、source_lon 和 domain_radius_km"
            )
        bumden_info = resolve_bumden_domain(
            selected_ice_data,
            source_lat,
            source_lon,
            domain_radius_km,
            geometry,
            processing,
            aggregate,
            min_domain_cells,
            data_source,
            hycom_processing,
        )
        bumden = float(bumden_info["BUMDEN_domain"])
    else:
        bumden = float(bumden_domain_ridges_per_km)
        bumden_info = {
            "BUMDEN_domain": bumden,
            "domain_stats": {},
            "ice_stats": {},
            "data_source": _normalise_ice_data_source(data_source),
        }
    if bumden < 0.0:
        raise ValueError("bumden_domain_ridges_per_km 必须大于等于 0")

    reflection, calculation = calculate_ice_reflection(
        frequency_hz,
        theta,
        bumden,
        geometry=geometry,
        hybrid_config=hybrid_config,
        force_model=force_model,
        enforce_passive=enforce_passive,
    )
    model_diag = calculation["model"]
    closure = calculation["closure"]
    valid = True
    if full_grazing_angle_domain:
        theta, reflection = cover_full_grazing_angle_domain(theta, reflection)
    path = write_trc(output_path, theta, reflection)
    diagnostics = {
        "bumden": bumden_info,
        "model": model_diag,
        "closure": {
            "passivity_clipped_count": closure["passivity_clipped_count"],
            "passivity_clipped_fraction": closure["passivity_clipped_fraction"],
        },
        "full_grazing_angle_domain": bool(full_grazing_angle_domain),
    }
    return TrcGenerationResult(
        path=path,
        frequency_hz=float(frequency_hz),
        bumden_domain_ridges_per_km=bumden,
        selected_model=str(model_diag.get("selected_model", "")),
        solver_valid=valid,
        angle_count=int(theta.size),
        passivity_clipped_count=int(closure["passivity_clipped_count"]),
        diagnostics=diagnostics,
    )


def generate_ice_reflection_trcs(
    output_dir: str | Path,
    frequencies_hz: Sequence[float],
    **kwargs: Any,
) -> list[TrcGenerationResult]:
    """在 output_dir 下为每个频率生成一个 .trc 文件。"""
    out = Path(output_dir)
    results: list[TrcGenerationResult] = []
    for frequency in sorted({float(f) for f in frequencies_hz if float(f) > 0.0}):
        path = out / f"domain_uniform_{freq_tag(frequency)}Hz.trc"
        results.append(generate_ice_reflection_trc(path, frequency, **kwargs))
    return results


def _cli() -> int:
    parser = argparse.ArgumentParser(description="生成 Bellhop .trc 冰下反射系数文件。")
    parser.add_argument("--ice-data", help="所选 HYCOM、neXtSIM 或 ICESat-2 .nc/.zip 文件")
    parser.add_argument(
        "--data-source",
        choices=["HYCOM", "NEXTSIM", "ICESAT2"],
        help=f"冰数据入口；不指定时使用文件顶部宏式设置 {ICE_REFLECTION_DATA_SOURCE}",
    )
    parser.add_argument("--source-lat", type=float)
    parser.add_argument("--source-lon", type=float)
    parser.add_argument("--domain-radius-km", type=float)
    parser.add_argument("--bumden-domain", type=float, help="直接使用该 BUMDEN，跳过冰数据处理")
    parser.add_argument("--freqs", nargs="+", type=float, required=True)
    parser.add_argument("--out-dir", default="ice_trc_output")
    parser.add_argument("--eta", type=float, default=8.2)
    parser.add_argument("--xi", type=float, default=5.1)
    parser.add_argument("--sound-speed", type=float, default=1500.0)
    parser.add_argument("--residual-tol", type=float, default=1e-4)
    parser.add_argument("--domain-aggregate", choices=["median", "mean", "p25", "p75"], default="median")
    parser.add_argument("--min-domain-cells", type=int, default=3)
    parser.add_argument("--window-cells", type=int, default=5)
    parser.add_argument("--background-percentile", type=float, default=25.0)
    parser.add_argument("--min-window-valid", type=int, default=5)
    parser.add_argument("--min-sic", type=float, default=0.15)
    parser.add_argument("--min-segments", type=int, default=1)
    parser.add_argument("--hycom-time-index", type=int, default=HYCOM_TIME_INDEX)
    parser.add_argument("--hycom-ice-density", type=float, default=HYCOM_ICE_DENSITY_KG_M3)
    parser.add_argument("--hycom-water-density", type=float, default=HYCOM_WATER_DENSITY_KG_M3)
    parser.add_argument("--hycom-bumden-scale", type=float, default=HYCOM_BUMDEN_SCALE)
    parser.add_argument("--hycom-no-sic-weighting", action="store_true")
    parser.add_argument("--blend-kb-start", type=float, default=16.0)
    parser.add_argument("--blend-kb-end", type=float, default=24.0)
    parser.add_argument("--hf-ppw", type=float, default=32.0)
    parser.add_argument("--hf-convergence-ppw", type=float, default=16.0)
    parser.add_argument("--hf-min-boundary-points", type=int, default=1024)
    parser.add_argument("--force-model", choices=["auto", "mathieu", "high_frequency"], default="auto")
    parser.add_argument("--no-passivity-clip", action="store_true")
    args = parser.parse_args()

    geometry = GeometryParams(
        eta_m=args.eta,
        xi_m=args.xi,
        sound_speed_m_s=args.sound_speed,
        residual_tol=args.residual_tol,
    )
    processing = IceProcessingConfig(
        window_cells=args.window_cells,
        background_percentile=args.background_percentile,
        min_window_valid=args.min_window_valid,
        min_sic=args.min_sic,
        min_segments=args.min_segments,
    )
    hybrid = HybridConfig(
        blend_kb_start=args.blend_kb_start,
        blend_kb_end=args.blend_kb_end,
        hf_config=HFConfig(
            points_per_wavelength=args.hf_ppw,
            convergence_points_per_wavelength=args.hf_convergence_ppw,
            min_boundary_points=args.hf_min_boundary_points,
        ),
    )
    hycom_processing = HycomProcessingConfig(
        window_cells=args.window_cells,
        background_percentile=args.background_percentile,
        min_window_valid=args.min_window_valid,
        min_sic=args.min_sic,
        time_index=args.hycom_time_index,
        ice_density_kg_m3=args.hycom_ice_density,
        water_density_kg_m3=args.hycom_water_density,
        weight_by_sic=not args.hycom_no_sic_weighting,
        bumden_scale=args.hycom_bumden_scale,
    )
    results = generate_ice_reflection_trcs(
        args.out_dir,
        args.freqs,
        bumden_domain_ridges_per_km=args.bumden_domain,
        ice_data=args.ice_data,
        source_lat=args.source_lat,
        source_lon=args.source_lon,
        domain_radius_km=args.domain_radius_km,
        geometry=geometry,
        processing=processing,
        aggregate=args.domain_aggregate,
        min_domain_cells=args.min_domain_cells,
        data_source=args.data_source,
        hycom_processing=hycom_processing,
        hybrid_config=hybrid,
        force_model=args.force_model,
        enforce_passive=not args.no_passivity_clip,
    )
    for result in results:
        print(
            f"{result.path} | f={result.frequency_hz:g} Hz | "
            f"BUMDEN={result.bumden_domain_ridges_per_km:.6g} ridges/km | "
            f"{result.selected_model}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
