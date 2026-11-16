"""ERA5/HYCOM 驱动的极区环境噪声分量工程预报。

这里输出的是环境场驱动的工程估算，不是从水听器数据反演得到的实测谱。
HYCOM/CICE 没有“冰裂声级”变量，因此冰裂分量使用冰速形变、冰浓度和
风冰相对速度构成的无量纲活动代理量。绝对量级需要用同区域水听器资料标定。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset, num2date

from .. import ice_reflection_trc
from .environment_noise import (
    component_levels_db,
    total_from_component_levels,
    wind_speed_mps_from_level,
)
from .hildebrand_wind_noise import hildebrand2021_psd_db


MIN_LEVEL_DB = -300.0
EARTH_RADIUS_M = 6371008.8


def _nc_path(path: str | Path) -> str:
    """兼容 Windows NetCDF C 库对中文绝对路径支持不完整的情况。"""
    try:
        return os.path.relpath(Path(path).resolve())
    except ValueError:
        return str(path)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(np.ma.filled(value, np.nan))
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _db_weight(level_db: np.ndarray, weight: float) -> np.ndarray:
    if not math.isfinite(float(weight)) or float(weight) <= 0.0:
        return np.full_like(np.asarray(level_db, dtype=np.float64), MIN_LEVEL_DB)
    return np.maximum(
        np.asarray(level_db, dtype=np.float64) + 10.0 * math.log10(float(weight)),
        MIN_LEVEL_DB,
    )


def _time_label(variable: Any, value: Any) -> str:
    units = getattr(variable, "units", "")
    calendar = getattr(variable, "calendar", "standard")
    if units:
        converted = num2date(value, units=units, calendar=calendar)
        # cftime 和 datetime 都提供这些字段。
        return (
            f"{int(converted.year):04d}-{int(converted.month):02d}-{int(converted.day):02d}"
            f"T{int(converted.hour):02d}:{int(converted.minute):02d}:{int(converted.second):02d}Z"
        )
    return datetime.fromtimestamp(int(value), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hour_key(value: Any) -> str:
    """把门户时间或标准化数据时间转换为 UTC 小时键。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError("时间不能为空。")
    parsed_text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(parsed_text)
    except ValueError as exc:
        raise ValueError(f"无法解析时间 {value!r}，应使用 yyyy-MM-dd HH:mm:ss。") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.strftime("%Y-%m-%dT%H")


def _select_time_index(variable: Any, requested_time: Any, label: str) -> tuple[int, str]:
    values = np.asarray(variable[:]).reshape(-1)
    if values.size < 1:
        raise ValueError(f"{label}时间轴为空。")
    labels = [_time_label(variable, value) for value in values]
    if requested_time in (None, ""):
        return 0, labels[0]
    requested_hour = _hour_key(requested_time)
    for index, time_label in enumerate(labels):
        if _hour_key(time_label) == requested_hour:
            return index, time_label
    raise ValueError(
        f"{label}没有请求小时 {requested_hour.replace('T', ' ')}:00:00 的数据；"
        f"可用时间范围为 {labels[0]} 至 {labels[-1]}。"
    )


def _axis_bracket(axis: np.ndarray, target: float) -> tuple[int, int, float]:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("环境数据坐标轴必须是一维且至少包含两个点。")
    ascending = values[0] < values[-1]
    work = values if ascending else values[::-1]
    upper = int(np.searchsorted(work, target, side="right"))
    upper = min(max(upper, 1), work.size - 1)
    lower = upper - 1
    low_value, high_value = float(work[lower]), float(work[upper])
    fraction = 0.0 if high_value == low_value else float(
        np.clip((target - low_value) / (high_value - low_value), 0.0, 1.0)
    )
    if ascending:
        return lower, upper, fraction
    return work.size - 1 - lower, work.size - 1 - upper, fraction


def _cyclic_lon_bracket(axis: np.ndarray, target: float) -> tuple[int, int, float]:
    longitude = np.mod(np.asarray(axis, dtype=np.float64), 360.0)
    order = np.argsort(longitude)
    sorted_lon = longitude[order]
    extended = np.concatenate((sorted_lon, [sorted_lon[0] + 360.0]))
    wanted = float(target) % 360.0
    if wanted < sorted_lon[0]:
        wanted += 360.0
    upper = int(np.searchsorted(extended, wanted, side="right"))
    upper = min(max(upper, 1), extended.size - 1)
    lower = upper - 1
    low_value, high_value = float(extended[lower]), float(extended[upper])
    fraction = 0.0 if high_value == low_value else float(
        np.clip((wanted - low_value) / (high_value - low_value), 0.0, 1.0)
    )
    return int(order[lower % order.size]), int(order[upper % order.size]), fraction


def _bilinear(values: np.ndarray, lat_fraction: float, lon_fraction: float) -> float:
    data = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    weights = np.asarray(
        [
            (1.0 - lat_fraction) * (1.0 - lon_fraction),
            (1.0 - lat_fraction) * lon_fraction,
            lat_fraction * (1.0 - lon_fraction),
            lat_fraction * lon_fraction,
        ],
        dtype=np.float64,
    )
    flat = data.reshape(-1)
    good = np.isfinite(flat) & (weights > 0.0)
    if not np.any(good):
        return math.nan
    return float(np.sum(flat[good] * weights[good]) / np.sum(weights[good]))


def _sample_regular_grid(
    dataset: Dataset,
    variable_name: str,
    time_index: int,
    lat_indices: tuple[int, int],
    lon_indices: tuple[int, int],
    lat_fraction: float,
    lon_fraction: float,
) -> float:
    variable = dataset.variables[variable_name]
    rows = np.asarray(lat_indices, dtype=np.int64)
    columns = np.asarray(lon_indices, dtype=np.int64)
    if variable.ndim == 3:
        plane = variable[int(time_index), rows, :][:, columns]
    elif variable.ndim == 2:
        plane = variable[rows, :][:, columns]
    else:
        raise ValueError(f"变量 {variable_name} 不是二维空间场。")
    return _bilinear(plane, lat_fraction, lon_fraction)


def _ice_deformation_rate(
    dataset: Dataset,
    time_index: int,
    lat_index: int,
    lon_index: int,
) -> float:
    """由 3x3 冰速邻域估算水平应变率，单位 s-1。"""
    latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
    longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
    i0, i1 = max(0, lat_index - 1), min(latitude.size - 1, lat_index + 1)
    j0, j1 = (lon_index - 1) % longitude.size, (lon_index + 1) % longitude.size
    if i0 == i1:
        return 0.0
    u = dataset.variables["siu"]
    v = dataset.variables["siv"]
    u_w = _finite(u[time_index, lat_index, j0], math.nan)
    u_e = _finite(u[time_index, lat_index, j1], math.nan)
    u_s = _finite(u[time_index, i0, lon_index], math.nan)
    u_n = _finite(u[time_index, i1, lon_index], math.nan)
    v_w = _finite(v[time_index, lat_index, j0], math.nan)
    v_e = _finite(v[time_index, lat_index, j1], math.nan)
    v_s = _finite(v[time_index, i0, lon_index], math.nan)
    v_n = _finite(v[time_index, i1, lon_index], math.nan)
    if not np.all(np.isfinite([u_w, u_e, u_s, u_n, v_w, v_e, v_s, v_n])):
        return 0.0
    dy = EARTH_RADIUS_M * math.radians(abs(float(latitude[i1] - latitude[i0])))
    lon_delta = abs(((float(longitude[j1] - longitude[j0]) + 180.0) % 360.0) - 180.0)
    dx = EARTH_RADIUS_M * max(abs(math.cos(math.radians(float(latitude[lat_index])))), 0.02) * math.radians(lon_delta)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    du_dx, du_dy = (u_e - u_w) / dx, (u_n - u_s) / dy
    dv_dx, dv_dy = (v_e - v_w) / dx, (v_n - v_s) / dy
    normal = du_dx - dv_dy
    shear = du_dy + dv_dx
    divergence = du_dx + dv_dy
    return float(math.sqrt(normal * normal + shear * shear + divergence * divergence))


@dataclass(frozen=True)
class PolarEnvironmentSample:
    longitude: float
    latitude: float
    time_utc: str
    wind_u10_mps: float
    wind_v10_mps: float
    wind_speed_mps: float
    ice_concentration: float
    ice_thickness_m: float
    ice_u_mps: float
    ice_v_mps: float
    ice_deformation_rate_s: float
    ice_age_years: float
    ridge_ratio: float
    snow_thickness_m: float
    young_ice_concentration: float
    multiyear_ice_concentration: float
    ice_albedo: float
    wind_file: str
    ice_file: str
    nextsim_file: str

    @property
    def wind_ice_relative_speed_mps(self) -> float:
        return float(math.hypot(self.wind_u10_mps - self.ice_u_mps, self.wind_v10_mps - self.ice_v_mps))


def sample_environment(
    wind_path: str | Path,
    ice_path: str | Path,
    nextsim_path: str | Path,
    longitude: float,
    latitude: float,
    requested_time: Any = None,
) -> PolarEnvironmentSample:
    """在请求点和请求小时插值 ERA5、HYCOM/CICE 和 neXtSIM 环境场。"""
    with Dataset(_nc_path(wind_path), "r") as wind:
        required = {"valid_time", "latitude", "longitude", "u10", "v10"}
        missing = sorted(required.difference(wind.variables))
        if missing:
            raise ValueError("ERA5 风场缺少变量：" + ", ".join(missing))
        wind_lat = np.asarray(wind.variables["latitude"][:], dtype=np.float64)
        wind_lon = np.asarray(wind.variables["longitude"][:], dtype=np.float64)
        wind_index, wind_time = _select_time_index(
            wind.variables["valid_time"], requested_time, "ERA5 10米风场"
        )
        wi0, wi1, wf_lat = _axis_bracket(wind_lat, float(latitude))
        wj0, wj1, wf_lon = _cyclic_lon_bracket(wind_lon, float(longitude))
        wind_u = _sample_regular_grid(
            wind, "u10", wind_index, (wi0, wi1), (wj0, wj1), wf_lat, wf_lon
        )
        wind_v = _sample_regular_grid(
            wind, "v10", wind_index, (wi0, wi1), (wj0, wj1), wf_lat, wf_lon
        )

    with Dataset(_nc_path(ice_path), "r") as ice:
        required = {"time", "lat", "lon", "sic", "sih", "siu", "siv"}
        missing = sorted(required.difference(ice.variables))
        if missing:
            raise ValueError("HYCOM/CICE 海冰场缺少变量：" + ", ".join(missing))
        ice_lat = np.asarray(ice.variables["lat"][:], dtype=np.float64)
        ice_lon = np.asarray(ice.variables["lon"][:], dtype=np.float64)
        ice_index, ice_time = _select_time_index(
            ice.variables["time"], requested_time, "HYCOM/CICE 海冰场"
        )
        ii0, ii1, if_lat = _axis_bracket(ice_lat, float(latitude))
        ij0, ij1, if_lon = _cyclic_lon_bracket(ice_lon, float(longitude))
        concentration = _sample_regular_grid(
            ice, "sic", ice_index, (ii0, ii1), (ij0, ij1), if_lat, if_lon
        )
        if not math.isfinite(concentration):
            concentration = 0.0
        if concentration > 1.5:
            concentration /= 100.0
        concentration = float(np.clip(concentration, 0.0, 1.0))
        thickness = _sample_regular_grid(
            ice, "sih", ice_index, (ii0, ii1), (ij0, ij1), if_lat, if_lon
        )
        ice_u = _sample_regular_grid(
            ice, "siu", ice_index, (ii0, ii1), (ij0, ij1), if_lat, if_lon
        )
        ice_v = _sample_regular_grid(
            ice, "siv", ice_index, (ii0, ii1), (ij0, ij1), if_lat, if_lon
        )
        if concentration <= 0.0:
            thickness, ice_u, ice_v = 0.0, 0.0, 0.0
        else:
            thickness = max(_finite(thickness), 0.0)
            ice_u, ice_v = _finite(ice_u), _finite(ice_v)
        centre_i = ii0 if if_lat < 0.5 else ii1
        centre_j = ij0 if if_lon < 0.5 else ij1
        deformation = (
            _ice_deformation_rate(ice, ice_index, centre_i, centre_j)
            if concentration > 0.0 else 0.0
        )

    with Dataset(_nc_path(nextsim_path), "r") as nextsim:
        required = {
            "time", "latitude", "longitude", "siconc", "sithick", "sisnthick",
            "siage", "si_ridge_ratio", "vxsi", "vysi", "siconc_young", "siconc_my",
        }
        missing = sorted(required.difference(nextsim.variables))
        if missing:
            raise ValueError("neXtSIM 海冰场缺少变量：" + ", ".join(missing))
        ns_lat = np.asarray(nextsim.variables["latitude"][:], dtype=np.float64)
        ns_lon = np.asarray(nextsim.variables["longitude"][:], dtype=np.float64)
        nextsim_index, nextsim_time = _select_time_index(
            nextsim.variables["time"], requested_time, "neXtSIM 小时场"
        )
        if not float(np.nanmin(ns_lat)) <= float(latitude) <= float(np.nanmax(ns_lat)):
            raise ValueError(
                f"请求纬度 {float(latitude):g}° 超出 neXtSIM 覆盖范围 "
                f"{float(np.nanmin(ns_lat)):g}° 到 {float(np.nanmax(ns_lat)):g}°。"
            )
        ni0, ni1, nf_lat = _axis_bracket(ns_lat, float(latitude))
        nj0, nj1, nf_lon = _cyclic_lon_bracket(ns_lon, float(longitude))

        def ns_value(name: str) -> float:
            return _sample_regular_grid(
                nextsim, name, nextsim_index, (ni0, ni1), (nj0, nj1), nf_lat, nf_lon
            )

        ns_concentration = ns_value("siconc")
        if math.isfinite(ns_concentration):
            concentration = float(np.clip(ns_concentration, 0.0, 1.0))
        ns_thickness = ns_value("sithick")
        if concentration > 0.0 and math.isfinite(ns_thickness):
            thickness = max(float(ns_thickness), 0.0)
        snow_thickness = max(_finite(ns_value("sisnthick")), 0.0)
        ice_age = max(_finite(ns_value("siage")), 0.0)
        ridge_ratio = float(np.clip(_finite(ns_value("si_ridge_ratio")), 0.0, 1.0))
        young_ice = float(np.clip(_finite(ns_value("siconc_young")), 0.0, 1.0))
        multiyear_ice = float(np.clip(_finite(ns_value("siconc_my")), 0.0, 1.0))
        ice_albedo = float(np.clip(_finite(ns_value("sialb"), math.nan), 0.0, 1.0)) \
            if "sialb" in nextsim.variables else math.nan
        ns_u = ns_value("vxsi")
        ns_v = ns_value("vysi")
        if concentration > 0.0 and np.all(np.isfinite([ns_u, ns_v])):
            ice_u, ice_v = float(ns_u), float(ns_v)
        # neXtSIM 约 3 km 网格比 HYCOM 更适合估计局地冰形变。
        ns_centre_i = ni0 if nf_lat < 0.5 else ni1
        ns_centre_j = nj0 if nf_lon < 0.5 else nj1
        # _ice_deformation_rate 使用的名称是 HYCOM 名称；neXtSIM 在此直接计算。
        i0, i1 = max(0, ns_centre_i - 1), min(ns_lat.size - 1, ns_centre_i + 1)
        j0, j1 = (ns_centre_j - 1) % ns_lon.size, (ns_centre_j + 1) % ns_lon.size
        vectors = [
            _finite(nextsim["vxsi"][nextsim_index, ns_centre_i, j0], math.nan),
            _finite(nextsim["vxsi"][nextsim_index, ns_centre_i, j1], math.nan),
            _finite(nextsim["vxsi"][nextsim_index, i0, ns_centre_j], math.nan),
            _finite(nextsim["vxsi"][nextsim_index, i1, ns_centre_j], math.nan),
            _finite(nextsim["vysi"][nextsim_index, ns_centre_i, j0], math.nan),
            _finite(nextsim["vysi"][nextsim_index, ns_centre_i, j1], math.nan),
            _finite(nextsim["vysi"][nextsim_index, i0, ns_centre_j], math.nan),
            _finite(nextsim["vysi"][nextsim_index, i1, ns_centre_j], math.nan),
        ]
        if i0 != i1 and np.all(np.isfinite(vectors)):
            dx = EARTH_RADIUS_M * max(
                abs(math.cos(math.radians(float(ns_lat[ns_centre_i])))), 0.02
            ) * math.radians(abs(((float(ns_lon[j1] - ns_lon[j0]) + 180.0) % 360.0) - 180.0))
            dy = EARTH_RADIUS_M * math.radians(abs(float(ns_lat[i1] - ns_lat[i0])))
            if dx > 0.0 and dy > 0.0:
                u_w, u_e, u_s, u_n, v_w, v_e, v_s, v_n = vectors
                du_dx, du_dy = (u_e - u_w) / dx, (u_n - u_s) / dy
                dv_dx, dv_dy = (v_e - v_w) / dx, (v_n - v_s) / dy
                ns_deformation = math.sqrt(
                    (du_dx - dv_dy) ** 2
                    + (du_dy + dv_dx) ** 2
                    + (du_dx + dv_dy) ** 2
                )
                if math.isfinite(ns_deformation):
                    deformation = float(ns_deformation)
    if _hour_key(wind_time) != _hour_key(ice_time):
        raise ValueError(f"ERA5 风场时次 {wind_time} 与 HYCOM/CICE 海冰时次 {ice_time} 不一致。")
    if _hour_key(nextsim_time) != _hour_key(wind_time):
        raise ValueError(
            f"neXtSIM 小时场时次 {nextsim_time} 与 ERA5/HYCOM 时次 {wind_time} 不一致。"
        )
    if not np.all(np.isfinite([wind_u, wind_v])):
        raise ValueError("请求点没有有效的 ERA5 10 米风矢量。")
    return PolarEnvironmentSample(
        longitude=((float(longitude) + 180.0) % 360.0) - 180.0,
        latitude=float(latitude),
        time_utc=wind_time,
        wind_u10_mps=float(wind_u),
        wind_v10_mps=float(wind_v),
        wind_speed_mps=float(math.hypot(wind_u, wind_v)),
        ice_concentration=concentration,
        ice_thickness_m=float(thickness),
        ice_u_mps=float(ice_u),
        ice_v_mps=float(ice_v),
        ice_deformation_rate_s=float(deformation),
        ice_age_years=float(ice_age),
        ridge_ratio=float(ridge_ratio),
        snow_thickness_m=float(snow_thickness),
        young_ice_concentration=float(young_ice),
        multiyear_ice_concentration=float(multiyear_ice),
        ice_albedo=float(ice_albedo),
        wind_file=Path(wind_path).name,
        ice_file=Path(ice_path).name,
        nextsim_file=Path(nextsim_path).name,
    )


def component_spectra_db(
    frequency_hz: Any,
    receiver_depth_m: float,
    sample: PolarEnvironmentSample,
    manual_levels: Mapping[str, int] | None = None,
    distributed_levels_db_fn: Any = None,
) -> dict[str, np.ndarray]:
    """计算开阔水、风-冰面和冰裂三类接收谱级工程估算。

    ``distributed_levels_db_fn`` 在数据驱动模式下提供风 / 降雨 / 航运三种
    网格建模谱级（dB，已在频率轴上求值），用于构造 ``openWaterNoiseLevel``；
    缺省为 None 时走经验公式路径。
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    sic = float(np.clip(sample.ice_concentration, 0.0, 1.0))

    if distributed_levels_db_fn is not None:
        # 数据驱动模式：风 / 降雨按未结冰比例加权，航运与手动模式一致不按
        # 密集度衰减，三者在线性功率域合成“非冰面噪声”。
        distributed = distributed_levels_db_fn(frequency)
        additions = {}
        if "wind" in distributed:
            additions["wind"] = _db_weight(distributed["wind"], 1.0 - sic)
        if "rain" in distributed:
            additions["rain"] = _db_weight(distributed["rain"], 1.0 - sic)
        if "shipping" in distributed:
            additions["shipping"] = distributed["shipping"]
        open_water = total_from_component_levels(additions)
    else:
        baseline = hildebrand2021_psd_db(frequency, sample.wind_speed_mps, receiver_depth_m)
        # 开阔水仅占未结冰面积，在线性功率域用开水比例加权。
        open_water = _db_weight(baseline, 1.0 - sic)

        if manual_levels is not None:
            # 冰区响应协议没有独立的航运和降雨返回字段，因此二者归入
            # openWaterNoiseLevel 这一“非冰面噪声”分量，并始终在线性功率域合成。
            manual_non_ice = component_levels_db(
                frequency,
                manual_levels["shipDensityLevel"],
                None,
                manual_levels["rainIntensity"],
            )
            additions = {"open_water": open_water, "shipping": manual_non_ice["shipping"]}
            if "rain" in manual_non_ice:
                additions["rain"] = _db_weight(manual_non_ice["rain"], 1.0 - sic)
            open_water = total_from_component_levels(additions)

    # 高密集度海冰显著削弱波浪/气泡型风噪声。这里用 1/4 风速灵敏度
    # （约对应公开观测中重冰区约 5logU、开水区约 20logU 的差异），并加入
    # 厚冰高频滚降。该项代表风驱动的冰面颗粒/摩擦背景，不包含离散冰裂。
    reduced_wind = max(1.0, math.pow(max(sample.wind_speed_mps, 1.0), 0.25))
    ice_baseline = hildebrand2021_psd_db(frequency, reduced_wind, receiver_depth_m)
    thickness_rolloff = -3.0 * max(sample.ice_thickness_m, 0.0) * np.log2(
        np.maximum(frequency / 500.0, 1.0)
    )
    marginal_activity = 1.0 + 3.0 * sic * (1.0 - sic)
    surface_particle_factor = (
        1.0
        + 2.0 * sample.ridge_ratio
        + 0.5 * min(sample.snow_thickness_m / 0.2, 2.0)
    )
    wind_and_ice = _db_weight(
        ice_baseline + thickness_rolloff - 6.0,
        sic * marginal_activity * surface_particle_factor,
    )

    # HYCOM/CICE 没有声学裂纹事件。以冰速空间形变、风冰相对运动和边缘冰区
    # 权重构造活动代理量；谱形在 500 Hz 附近最强，并限制为宽带工程估算。
    deformation_index = np.clip(sample.ice_deformation_rate_s / 1.0e-6, 0.0, 20.0)
    relative_index = np.clip(sample.wind_ice_relative_speed_mps / 10.0, 0.0, 4.0)
    brittle_age_factor = 1.0 + min(sample.ice_age_years, 3.0) / 3.0
    crack_activity = sic * brittle_age_factor * (
        0.05
        + deformation_index
        + 2.0 * sample.ridge_ratio
        + relative_index * (0.25 + 2.0 * (1.0 - sic))
    )
    crack_shape = -12.0 * np.abs(np.log2(np.maximum(frequency, 1.0) / 500.0))
    crack_reference = 58.0 + 10.0 * math.log10(max(crack_activity, 1.0e-12))
    ice_crack = np.maximum(crack_reference + crack_shape, MIN_LEVEL_DB)
    if sic <= 0.0:
        ice_crack[:] = MIN_LEVEL_DB

    return {
        "windAndIceNoiseLevel": wind_and_ice,
        "IceCrackNoiseLevel": ice_crack,
        "openWaterNoiseLevel": open_water,
    }


def _total_at_frequency(
    frequency_hz: float,
    depth_m: float,
    sample: PolarEnvironmentSample,
    manual_levels: Mapping[str, int] | None = None,
    distributed_levels_db_fn: Any = None,
) -> float:
    components = component_spectra_db(
        np.asarray([frequency_hz]),
        depth_m,
        sample,
        manual_levels,
        distributed_levels_db_fn,
    )
    return float(total_from_component_levels(components)[0])


def ice_propagation_directivity_db(
    frequency_hz: float,
    receiver_depth_m: float,
    count: int,
    sample: PolarEnvironmentSample,
    bumden_domain_ridges_per_km: float,
    manual_levels: Mapping[str, int] | None = None,
    distributed_levels_db_fn: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用冰面复反射幅相计算垂直面相干传播指向性。

    负角表示朝向冰面的上行方向，正角表示下行方向。计算使用与 Bellhop
    ``F`` 顶边界相同的 Burke–Twersky 半椭圆冰脊复反射系数，并叠加直达场
    与冰面镜像场。粗糙冰面造成的非相干能量由冰脊率和海冰密集度约束，避免
    把有限带宽的环境噪声误算成数学上的无限深零陷。
    """
    signed_angle = np.linspace(-90.0, 90.0, int(count), dtype=np.float64)
    reflection_angle = np.clip(np.abs(signed_angle), 0.5, 89.5)
    # 与落盘给 Bellhop 的 .trc 使用同一标准角网格；响应采样点再对复幅相插值。
    # 这样 directivityCount 只改变输出分辨率，不会把高频积分规模放大到不可控。
    model_angle = ice_reflection_trc.angle_grid()
    model_reflection, _diagnostics = ice_reflection_trc.calculate_ice_reflection(
        float(frequency_hz),
        model_angle,
        float(bumden_domain_ridges_per_km),
    )
    reflection_magnitude = np.interp(
        reflection_angle,
        model_angle,
        np.abs(model_reflection),
    )
    reflection_phase = np.interp(
        reflection_angle,
        model_angle,
        np.unwrap(np.angle(model_reflection)),
    )
    reflection = reflection_magnitude * np.exp(1j * reflection_phase)

    wave_number = 2.0 * math.pi * float(frequency_hz) / 1500.0
    vertical_phase = (
        -2.0
        * wave_number
        * float(receiver_depth_m)
        * np.sin(np.deg2rad(signed_angle))
    )
    coherent_power = np.square(np.abs(1.0 + reflection * np.exp(1j * vertical_phase)))
    incoherent_power = 1.0 + np.square(np.abs(reflection))

    sic = float(np.clip(sample.ice_concentration, 0.0, 1.0))
    ridge_ratio = float(np.clip(sample.ridge_ratio, 0.0, 1.0))
    coherent_fraction = float(np.clip(
        0.88 - 0.45 * ridge_ratio - 0.15 * (1.0 - sic),
        0.25,
        0.90,
    ))
    projection = 0.25 + 0.75 * np.abs(np.sin(np.deg2rad(signed_angle)))
    gain_power = projection * (
        coherent_fraction * coherent_power
        + (1.0 - coherent_fraction) * incoherent_power
    )
    gain_power = np.maximum(gain_power, 1.0e-12)
    gain_power /= max(float(np.mean(gain_power)), 1.0e-12)

    reference_level = _total_at_frequency(
        float(frequency_hz),
        float(receiver_depth_m),
        sample,
        manual_levels,
        distributed_levels_db_fn,
    )
    level_db = reference_level + 10.0 * np.log10(gain_power)
    return signed_angle, level_db


def build_response(
    params: Mapping[str, Any],
    sample: PolarEnvironmentSample,
    *,
    ice_reflection_bumden_ridges_per_km: float,
    distributed_levels_db_fn: Any = None,
    directivity_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按门户约定生成深度、频谱和指向性三段响应。"""
    frequency = np.linspace(
        float(params["minFreq"]),
        float(params["maxFreq"]),
        int(params["numPoints"]),
        dtype=np.float64,
    )
    receiver_depth = float(params["receiverDepthM"])
    calc_frequency = float(params["calcFrequency"])
    manual_levels = params.get("manualLevels")
    if manual_levels is not None:
        manual_wind_speed = wind_speed_mps_from_level(manual_levels["windSpeed"])
        current_speed = max(sample.wind_speed_mps, 0.0)
        if current_speed > 1.0e-9:
            scale = manual_wind_speed / current_speed
            wind_u, wind_v = sample.wind_u10_mps * scale, sample.wind_v10_mps * scale
        else:
            wind_u, wind_v = manual_wind_speed, 0.0
        sample = replace(
            sample,
            wind_u10_mps=float(wind_u),
            wind_v10_mps=float(wind_v),
            wind_speed_mps=float(manual_wind_speed),
        )
    components = component_spectra_db(
        frequency, receiver_depth, sample, manual_levels, distributed_levels_db_fn
    )
    total = total_from_component_levels(components)

    depth = np.linspace(
        1.0,
        float(params["profileDepthMaxM"]),
        int(params["profileDepthCount"]),
        dtype=np.float64,
    )
    depth_level = np.asarray(
        [
            _total_at_frequency(
                calc_frequency, float(value), sample, manual_levels, distributed_levels_db_fn
            )
            for value in depth
        ],
        dtype=np.float64,
    )

    if directivity_override is None:
        grazing, directivity = ice_propagation_directivity_db(
            calc_frequency,
            receiver_depth,
            int(params["directivityCount"]),
            sample,
            ice_reflection_bumden_ridges_per_km,
            manual_levels,
            distributed_levels_db_fn,
        )
        normalized_directivity = directivity - float(np.max(directivity))
        directivity_metadata = {
            "method": "legacy_ice_surface_image_interference",
        }
    else:
        grazing = np.asarray(directivity_override["grazingAngle"], dtype=np.float64)
        directivity = np.asarray(directivity_override["absNoiseLevel"], dtype=np.float64)
        normalized_directivity = np.asarray(
            directivity_override["normalizedLevelDb"], dtype=np.float64
        )
        directivity_metadata = dict(directivity_override.get("metadata") or {})
        directivity_metadata["componentSourceCounts"] = dict(
            directivity_override.get("componentSourceCounts") or {}
        )
    peak_index = int(np.argmax(directivity))
    notch_index = int(np.argmin(directivity))

    rounded_frequency = np.round(frequency, 6).tolist()
    response = {
        "depthProfile": {
            "calcFrequency": round(calc_frequency, 6),
            "size": int(depth.size),
            "depth": np.round(depth, 6).tolist(),
            "noiseLevel": np.round(depth_level, 6).tolist(),
        },
        "noiseSpectrum": {
            "size": int(frequency.size),
            "frequency": rounded_frequency,
            "totalNoiseLevel": np.round(total, 6).tolist(),
            "windAndIceNoiseLevel": np.round(components["windAndIceNoiseLevel"], 6).tolist(),
            "IceCrackNoiseLevel": np.round(components["IceCrackNoiseLevel"], 6).tolist(),
            "openWaterNoiseLevel": np.round(components["openWaterNoiseLevel"], 6).tolist(),
        },
        "noiseDirectivity": {
            "peakGrazingAngle": round(float(grazing[peak_index]), 6),
            "notchGrazingAngle": round(float(grazing[notch_index]), 6),
            "size": int(grazing.size),
            "grazingAngle": np.round(grazing, 6).tolist(),
            "absNoiseLevel": np.round(directivity, 6).tolist(),
            "normalizedLevelDb": np.round(normalized_directivity, 6).tolist(),
            "metadata": directivity_metadata,
        },
    }
    return response
