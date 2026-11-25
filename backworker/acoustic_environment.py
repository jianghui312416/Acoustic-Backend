# -*- coding: utf-8 -*-
"""三个声学模型共用的海洋环境获取与剖面准备入口。

该模块把地形、温盐/声速、海况和海底声学数据的读取封装成统一环境对象。
Bellhop、Kraken 和 RAM 只消费这里产出的声源上下文或二维方位剖面，不再各自
直接组织环境数据读取流程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


ProgressCallback = Callable[[str, str], None]


class AcousticEnvironmentSource(Protocol):
    """公共环境模块需要的数据源能力。"""

    def resolve_topography_file(self, params: Mapping[str, Any], config: Mapping[str, Any]) -> Path: ...
    def open_topography(self, path: Path) -> Any: ...
    def build_source_grids(
        self,
        points: Sequence[Mapping[str, float]],
        params: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]: ...
    def open_data_access(self, params: Mapping[str, Any], config: Mapping[str, Any]) -> Any: ...
    def prepare_shared_environment(
        self,
        data_access: Any,
        topography: Any,
        source_grids: Sequence[Mapping[str, Any]],
        settings: Mapping[str, Any],
    ) -> Any: ...


BOTTOM_GEO_FIELDS = (
    "compressional_speed_mps",
    "shear_speed_mps",
    "density_g_cm3",
    "compressional_attenuation",
    "shear_attenuation",
)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """读取必需的映射配置。"""
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"公共环境配置字段 {key!r} 必须是 JSON 对象。")
    return value


def _boolean(value: Any) -> bool:
    """把常用文本或数值表示转换为布尔值。"""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalise_longitude(value: float) -> float:
    """把经度转换到 -180 到 180 度范围。"""
    return (float(value) + 180.0) % 360.0 - 180.0


def get_bottom_geo_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """返回公共环境中唯一的海底声学配置。"""
    common = _mapping(config, "ACOUSTIC_ENVIRONMENT")
    return _mapping(common, "BOTTOM_GEOACOUSTICS")


def prepare_common_task(
    supplied: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    max_sources: int = 1,
) -> tuple[Dict[str, Any], list[Dict[str, float]]]:
    """合并并校验与具体声学求解器无关的任务参数。"""
    defaults = config.get("TASK_DEFAULTS")
    if not isinstance(defaults, Mapping):
        raise ValueError("TASK_DEFAULTS 必须是 JSON 对象。")
    params = dict(defaults)
    # config.TASK_DEFAULTS兜底，传入的参数supplied覆盖
    params.update(supplied)

    for key, label in (
        ("freq_hz", "频率"),
        ("source_depth_m", "声源深度"),
        ("receiver_depth_max_m", "最大接收深度"),
        ("radius_km", "传播半径"),
    ):
        try:
            value = float(params[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label}必须是有效数值。") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}必须为正数。")

    try:
        latitude = float(params["center_lat"])
        longitude = float(params["center_lon"])
        grid_count = int(round(float(params["n_grid"])))
        section_count = int(round(float(params["section_count"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("声源位置、环境网格或方位数量参数无效。") from exc
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("纬度输入不合法，必须位于 -90 到 90°。")
    if not math.isfinite(longitude) or not -360.0 <= longitude <= 360.0:
        raise ValueError("经度输入不合法，必须位于 -360 到 360°。")
    if grid_count < 3:
        raise ValueError("局部环境网格点数至少为 3。")
    minimum_sections = 1 if str(params.get("calculation_mode", "3d")).lower() == "2d" else 3
    if section_count < minimum_sections:
        raise ValueError(f"方位切面数量至少为 {minimum_sections}。")

    requested_points = params.get("selection_points")
    raw_points = requested_points if isinstance(requested_points, list) and requested_points else [
        {"lat": latitude, "lon": longitude}
    ]
    if len(raw_points) > max_sources:
        raise ValueError(f"声源数 {len(raw_points)} 超过当前模型限制 {max_sources}。")
    points = []
    for point in raw_points:
        if not isinstance(point, Mapping):
            raise ValueError("每个声源点必须是包含 lat/lon 的对象。")
        point_lat = float(point["lat"])
        point_lon = normalise_longitude(float(point["lon"]))
        if not -90.0 <= point_lat <= 90.0:
            raise ValueError("声源点纬度必须位于 -90 到 90°。")
        points.append({"lat": point_lat, "lon": point_lon})

    params["n_grid"] = grid_count
    params["section_count"] = section_count
    params["selection_points"] = points
    params["selection_point_count"] = len(points)
    params["source_mode"] = "independent_multi_source" if len(points) > 1 else "single_source"
    params["center_lat"] = points[0]["lat"]
    params["center_lon"] = points[0]["lon"]
    return params, points


def environment_settings(params: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """读取与 Bellhop、Kraken、RAM 均无关的环境采样配置。"""
    common = _mapping(config, "ACOUSTIC_ENVIRONMENT")
    bottom = _mapping(common, "BOTTOM")
    bottom_geo = get_bottom_geo_config(config)
    depth_grid = [float(value) for value in config.get("DEPTH_GRID_M", [])]
    if len(depth_grid) < 4 or np.any(np.diff(depth_grid) <= 0.0):
        raise ValueError("DEPTH_GRID_M 必须是至少含 4 个严格递增深度的数组。")

    padding_km = float(common.get("PADDING_KM", 100.0))
    profile_extra_km = float(common.get("PROFILE_EXTRA_KM", 25.0))
    depth_guard_m = float(common.get("DEPTH_GUARD_M", 10.0))
    if padding_km <= profile_extra_km or profile_extra_km < 0.0:
        raise ValueError("公共环境 PADDING_KM 必须大于 PROFILE_EXTRA_KM。")
    if depth_guard_m <= 0.0:
        raise ValueError("公共环境深度保护量必须为正数。")

    configured_mode = str(common.get("SSP_MODE", "range_dependent")).strip().lower()
    if params.get("is_water_changed") is not None:
        ssp_mode = "range_dependent" if _boolean(params.get("is_water_changed")) else "fixed"
    else:
        ssp_mode = configured_mode
    if ssp_mode not in {"fixed", "mean", "range_dependent"}:
        raise ValueError("公共环境 SSP_MODE 必须是 fixed、mean 或 range_dependent。")

    bottom_fixed = {
        "compressional_speed_mps": float(bottom.get("COMPRESSIONAL_SPEED_MPS", 1600.0)),
        "shear_speed_mps": float(bottom.get("SHEAR_SPEED_MPS", 0.0)),
        "density_g_cm3": float(bottom.get("DENSITY_G_CM3", 1.8)),
        "compressional_attenuation": float(bottom.get("COMPRESSIONAL_ATTENUATION", 0.8)),
        "shear_attenuation": float(bottom.get("SHEAR_ATTENUATION", 0.0)),
    }
    variables = {}
    scales = {}
    offsets = {}
    ranges = {}
    raw_variables = bottom_geo.get("NETCDF_VARIABLES", {})
    raw_scales = bottom_geo.get("SCALE", {})
    raw_offsets = bottom_geo.get("OFFSET", {})
    raw_ranges = bottom_geo.get("VALID_RANGES", {})
    for name in BOTTOM_GEO_FIELDS:
        names = raw_variables.get(name, [])
        variables[name] = [str(value) for value in ([names] if isinstance(names, str) else names)]
        scales[name] = float(raw_scales.get(name, 1.0))
        offsets[name] = float(raw_offsets.get(name, 0.0))
        bounds = raw_ranges.get(name, [0.0, 1.0e9])
        if not isinstance(bounds, Sequence) or isinstance(bounds, str) or len(bounds) != 2:
            raise ValueError(f"公共环境 VALID_RANGES.{name} 必须包含两个数值。")
        ranges[name] = [float(bounds[0]), float(bounds[1])]

    data_sources = config.get("DATA_SOURCES", {})
    bottom_source = data_sources.get("bottom_geoacoustics", {}) if isinstance(data_sources, Mapping) else {}
    source_mode = str(bottom_source.get("mode", "file")).strip().lower() if isinstance(bottom_source, Mapping) else "file"
    interpolation = (
        str(bottom_source.get("interpolation", "nearest" if source_mode == "postgres" else "linear"))
        .strip()
        .lower()
        if isinstance(bottom_source, Mapping)
        else "linear"
    )
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("海底声学 interpolation 必须是 linear 或 nearest。")
    bottom_geo_enabled = _boolean(bottom_geo.get("ENABLED", False))

    return {
        "padding_km": padding_km,
        "profile_extra_km": profile_extra_km,
        "depth_guard_m": depth_guard_m,
        "depth_grid_m": depth_grid,
        "ssp_mode": ssp_mode,
        "bottom_fixed": bottom_fixed,
        "bottom_fixed_source": "ACOUSTIC_ENVIRONMENT.BOTTOM fixed values",
        "bottom_geo_mode": f"{source_mode}_long" if bottom_geo_enabled else "fixed",
        "bottom_geo_long": bottom_geo_enabled,
        "bottom_geo_interpolation": interpolation,
        "bottom_geo_required_fields": [str(value) for value in bottom_geo.get("REQUIRED_FIELDS", [])],
        "bottom_geo_variables": variables,
        "bottom_geo_scale": scales,
        "bottom_geo_offset": offsets,
        "bottom_geo_valid_ranges": ranges,
        "ice_reflection_bumden_domain": params.get("ice_reflection_bumden_domain"),
    }


def bearing_angles(params: Mapping[str, Any]) -> np.ndarray:
    """根据公共任务参数生成去重的二维剖面方位角。"""
    minimum = 1 if str(params.get("calculation_mode", "3d")).lower() == "2d" else 3
    count = max(minimum, int(params["section_count"]))
    start = float(params["beta_min"])
    stop = float(params["beta_max"])
    if count == 1:
        return np.asarray([start % 360.0], dtype=np.float64)
    span = stop - start
    if abs(span) >= 359.0 - 1.0e-9:
        direction = -1.0 if span < 0.0 else 1.0
        values = start + direction * np.linspace(0.0, 360.0, count, endpoint=False)
    else:
        values = np.linspace(start, stop, count)
    return np.mod(values, 360.0).astype(np.float64)


@dataclass
class PreparedAcousticEnvironment:
    """一次环境读取产生的共享数据及其生命周期。"""

    params: Dict[str, Any]
    config: Mapping[str, Any]
    settings: Dict[str, Any]
    points: Sequence[Mapping[str, float]]
    topography_file: Path
    topography: Any
    source_grids: Sequence[Mapping[str, Any]]
    shared_environment: Any
    data_access: Any
    source: AcousticEnvironmentSource

    def close(self) -> None:
        """释放数据库或外部数据源连接。"""
        self.data_access.close()

    def cache_info(self) -> Dict[str, Any]:
        """返回共享环境读取统计，供任务清单和进度信息使用。"""
        shared = self.shared_environment
        return {
            "strategy": "一次读取并集，再精确切回各局部网格",
            "local_grid_sample_count": shared.point_count,
            "temperature_cell_count": shared.temperature_cell_count,
            "salinity_cell_count": shared.salinity_cell_count,
            "bottom_geoacoustics_mode": shared.bottom_geo_mode,
            "bottom_geoacoustics_source": shared.bottom_geo_source,
        }

    def build_bellhop_source_context(
        self,
        source_offset: int,
        work_dir: Path,
        progress: ProgressCallback,
        context_builder: Callable[..., Dict[str, Any]],
    ) -> Dict[str, Any]:
        """为 Bellhop 构造包含求解器清单和并行信息的完整上下文。"""
        point = self.points[source_offset]
        grid = self.source_grids[source_offset]
        prepared = (grid,) + self.shared_environment.for_source(
            source_offset,
            grid,
            float(self.params["source_depth_m"]),
        )
        return context_builder(
            source_offset + 1,
            float(point["lat"]),
            normalise_longitude(float(point["lon"])),
            work_dir,
            self.params,
            self.config,
            self.settings,
            self.topography,
            float(self.params["radius_km"]),
            progress,
            prepared,
            self.data_access,
        )

    def build_common_source_context(self, source_offset: int) -> Dict[str, Any]:
        """构造 Kraken/RAM 使用的求解器无关环境上下文。"""
        point = self.points[source_offset]
        grid = self.source_grids[source_offset]
        environment, bathymetry, altimetry, bottom_geoacoustics = self.shared_environment.for_source(
            source_offset,
            grid,
            float(self.params["source_depth_m"]),
        )
        model_depth = float(np.asarray(environment["depth_m"], dtype=np.float64)[-1])
        source_depth = float(self.params["source_depth_m"])
        depth_guard = float(self.settings["depth_guard_m"])
        if model_depth <= source_depth + depth_guard:
            raise ValueError("环境数据最大深度无法覆盖声源深度。")
        receiver_depth = min(
            float(self.params["receiver_depth_max_m"]),
            model_depth - depth_guard,
        )
        return {
            "source_index": source_offset + 1,
            "source_lat": float(point["lat"]),
            "source_lon": normalise_longitude(float(point["lon"])),
            "settings": dict(self.settings),
            "grid": grid,
            "environment": environment,
            "bathymetry": bathymetry,
            "altimetry": altimetry,
            "bottom_geoacoustics": bottom_geoacoustics,
            "calculation_radius_km": float(self.params["radius_km"]),
            "tl_depth_max_m": receiver_depth,
            "bearings": bearing_angles(self.params),
        }

    def sample_sections(
        self,
        context: Mapping[str, Any],
        calculation_range_km: float,
        profile_count: int,
        extra_range_km: float = 0.0,
    ) -> list[Dict[str, Any]]:
        """按方位生成二维剖面。

        ``profile_count`` 必须由当前求解器自己的配置提供，公共环境不再
        保存任何模型的距离采样点数。
        """
        count = int(profile_count)
        if count < 3:
            raise ValueError("二维环境剖面的距离采样点数至少为 3。")
        sections = []
        for bearing in context["bearings"]:
            section = sample_2d_slice(
                context["grid"],
                context["environment"],
                context["bathymetry"],
                context["altimetry"],
                context["bottom_geoacoustics"],
                float(bearing),
                float(calculation_range_km),
                count,
                extra_range_km=float(extra_range_km),
            )
            sections.append(
                apply_ssp_mode(
                    section,
                    self.settings["ssp_mode"],
                )
            )
        return sections


def range_samples(
    calculation_range_km: float,
    sample_count: int,
    extra_range_km: float = 0.0,
) -> np.ndarray:
    """生成距离轴。

    ``sample_count`` 只描述 ``[0, calculation_range_km]`` 内的采样点数，
    并包含 0 和计算半径两个端点。扩展区沿用半径内的采样间距追加点；若扩展
    终点不落在等间距网格上，再补一个精确终点。
    """
    calculation_range = float(calculation_range_km)
    extra_range = float(extra_range_km)
    count = int(sample_count)
    if not math.isfinite(calculation_range) or calculation_range <= 0.0:
        raise ValueError("计算半径必须是正的有限数值。")
    if count < 3:
        raise ValueError("计算半径内的距离采样点数至少为 3。")
    if not math.isfinite(extra_range) or extra_range < 0.0:
        raise ValueError("距离扩展量必须是非负有限数值。")

    requested = np.linspace(0.0, calculation_range, count, dtype=np.float64)
    if extra_range == 0.0:
        return requested

    spacing = calculation_range / (count - 1)
    extended_end = calculation_range + extra_range
    full_step_count = int(math.floor(extra_range / spacing))
    extension = calculation_range + spacing * np.arange(
        1,
        full_step_count + 1,
        dtype=np.float64,
    )
    tolerance = max(1.0, abs(extended_end)) * 1.0e-12
    extension = extension[extension < extended_end - tolerance]
    return np.concatenate((requested, extension, np.asarray([extended_end], dtype=np.float64)))


def sample_2d_slice(
    grid: Mapping[str, Any],
    environment: Mapping[str, Any],
    bathymetry: Mapping[str, Any],
    altimetry: Mapping[str, Any],
    bottom_geoacoustics: Mapping[str, Any],
    bearing_deg: float,
    calculation_range_km: float,
    profile_count: int,
    extra_range_km: float = 0.0,
) -> Dict[str, Any]:
    """从区域环境场采样一个与求解器无关的二维方位剖面。"""
    ranges = range_samples(calculation_range_km, profile_count, extra_range_km)
    angle = math.radians(float(bearing_deg))
    x = ranges * math.sin(angle)
    y = ranges * math.cos(angle)
    points = np.column_stack([y, x])
    axes = (
        np.asarray(grid["y_rel_axis_km"], dtype=np.float64),
        np.asarray(grid["x_rel_axis_km"], dtype=np.float64),
    )

    def sample(values: np.ndarray, method: str = "linear") -> np.ndarray:
        """沿当前方位对一个二维环境字段进行插值。"""
        interpolator = RegularGridInterpolator(
            axes,
            np.asarray(values, dtype=np.float64),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        return interpolator(points)

    speed = np.vstack([sample(layer) for layer in environment["sound_speed_mps"]])
    for row in speed:
        finite = row[np.isfinite(row)]
        row[~np.isfinite(row)] = float(np.mean(finite)) if finite.size else 1500.0
    bottom = sample(bathymetry["depth_m"])
    surface_depth = -sample(altimetry["elevation_m"])
    land = sample(np.asarray(bathymetry["land_mask"], dtype=np.float64), method="nearest") >= 0.5
    model_depth = float(np.asarray(environment["depth_m"], dtype=np.float64)[-1])
    bottom = np.clip(bottom, 1.0, model_depth)
    surface_depth = np.where(np.isfinite(surface_depth), surface_depth, 0.0)
    surface_depth = np.clip(surface_depth, 0.0, bottom - 0.5)
    bottom_interpolation = str(bottom_geoacoustics.get("interpolation", "linear"))
    bottom_fields = {
        name: sample(np.asarray(values, dtype=np.float64), method=bottom_interpolation)
        for name, values in bottom_geoacoustics["fields"].items()
    }
    for values in bottom_fields.values():
        fallback = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
        values[~np.isfinite(values)] = fallback
    return {
        "bearing_deg": float(bearing_deg),
        "range_km": ranges,
        "x_km": x,
        "y_km": y,
        "sound_speed_mps": speed,
        "bottom_depth_m": bottom,
        "surface_depth_m": surface_depth,
        "land_mask": land,
        "bottom_geoacoustics": {
            "mode": str(bottom_geoacoustics["mode"]),
            "source": str(bottom_geoacoustics["source"]),
            "interpolation": bottom_interpolation,
            "loaded_fields": list(bottom_geoacoustics.get("loaded_fields", [])),
            "fallback_fields": list(bottom_geoacoustics.get("fallback_fields", [])),
            "fields": bottom_fields,
        },
    }


def apply_ssp_mode(section: Dict[str, Any], ssp_mode: str) -> Dict[str, Any]:
    """把固定 SSP 模式转换为全距离复用声源位置剖面。"""
    if ssp_mode != "fixed":
        return section
    ranges = np.asarray(section["range_km"], dtype=np.float64)
    sound_speed = np.asarray(section["sound_speed_mps"], dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("固定 SSP 的距离轴必须是一维非空数组。")
    if sound_speed.ndim != 2 or sound_speed.shape[1] != ranges.size:
        raise ValueError("固定 SSP 声速网格必须是 (depth, range) 形状。")
    fixed_section = dict(section)
    fixed_section["sound_speed_mps"] = np.repeat(sound_speed[:, :1], ranges.size, axis=1)
    return fixed_section


def acquire(
    params: Dict[str, Any],
    config: Mapping[str, Any],
    settings: Dict[str, Any],
    points: Sequence[Mapping[str, float]],
    source: AcousticEnvironmentSource,
) -> PreparedAcousticEnvironment:
    """一次性读取所有声源需要的共享环境数据。"""
    from .netcdf_access_lock import locked as netcdf_read_locked

    # 所有模型统一从这里进入环境数据读取。锁由 dispatcher 父进程创建并注入
    # 四个声场 worker，只保护 NetCDF/HDF5 打开和复制到内存的阶段。
    with netcdf_read_locked():
        topography_file = source.resolve_topography_file(params, config)
        topography_path = Path(str(topography_file))
        if not topography_path.is_file():
            raise FileNotFoundError(f"地形文件不存在：{topography_file or '(空路径)'}")

        topography = source.open_topography(topography_path)
        source_grids = source.build_source_grids(points, params, settings)
        data_access = source.open_data_access(params, config)
        try:
            data_access.validate()
            shared_environment = source.prepare_shared_environment(
                data_access,
                topography,
                source_grids,
                settings,
            )
        except Exception:
            data_access.close()
            raise

    return PreparedAcousticEnvironment(
        params=params,
        config=config,
        settings=settings,
        points=points,
        topography_file=topography_path,
        topography=topography,
        source_grids=source_grids,
        shared_environment=shared_environment,
        data_access=data_access,
        source=source,
    )
