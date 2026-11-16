"""ERA5 风噪声源经声传播后在指定接收点的功率谱计算。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

EARTH_RADIUS_M = 6371008.8
RAIN_MODEL_MIN_FREQUENCY_HZ = 500.0
RAIN_MODEL_MAX_FREQUENCY_HZ = 6400.0
RAIN_SOURCE_REFERENCE_SOUND_SPEED_MPS = 1500.0


def nc_path(path):
    """兼容 Windows NetCDF C 库不能直接打开中文绝对路径的情况。"""
    try:
        return os.path.relpath(Path(path).resolve())
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class WindSourceSelection:
    """一次接收点计算实际使用的局部海面风噪声源。"""

    product_path: Path
    time_index: int
    time_utc: str
    receiver_lon: float
    receiver_lat: float
    radius_km: float
    source_depth_m: float
    longitude: np.ndarray
    latitude: np.ndarray
    distance_km: np.ndarray
    bearing_deg: np.ndarray
    wind_speed_mps: np.ndarray
    cell_area_m2: np.ndarray
    reference_transfer_integral_m2: float

    @property
    def source_count(self) -> int:
        return int(self.distance_km.size)


@dataclass(frozen=True)
class RainSourceSelection:
    """ERA5 液态降水网格在接收点附近的有效声源。"""

    product_path: Path
    time_index: int
    time_utc: str
    receiver_lon: float
    receiver_lat: float
    radius_km: float
    source_depth_m: float
    longitude: np.ndarray
    latitude: np.ndarray
    distance_km: np.ndarray
    bearing_deg: np.ndarray
    rain_rate_mm_h: np.ndarray
    cell_area_m2: np.ndarray
    reference_transfer_integral_m2: float

    @property
    def source_count(self) -> int:
        return int(self.distance_km.size)


@dataclass(frozen=True)
class ShippingSourceSelection:
    """AIS/GFW 稀疏航运网格在接收点附近的有效声源。"""

    product_path: Path
    time_index: int
    time_utc: str
    receiver_lon: float
    receiver_lat: float
    radius_km: float
    source_depth_m: float
    longitude: np.ndarray
    latitude: np.ndarray
    distance_km: np.ndarray
    bearing_deg: np.ndarray
    frequency_hz: np.ndarray
    cell_source_power_linear: np.ndarray

    @property
    def source_count(self) -> int:
        return int(self.distance_km.size)


@dataclass(frozen=True)
class ReceiverWindSample:
    """插值到单个接收点位置的 ERA5 10 m 风场。"""

    product_path: Path
    time_index: int
    time_utc: str
    receiver_lon: float
    receiver_lat: float
    u10_mps: float
    v10_mps: float
    wind_speed_mps: float
    interpolation: str


def uniform_polar_wind_selection(
    selection: WindSourceSelection,
    wind_speed_mps: float,
    *,
    sector_count: int,
    radial_step_km: float,
) -> WindSourceSelection:
    """把局部 ERA5 风速重采样为论文式规则扇环均匀风面源。"""
    sectors = int(sector_count)
    radial_step = float(radial_step_km)
    speed = float(wind_speed_mps)
    if sectors < 4:
        raise ValueError("论文风面源方位数不能小于 4。")
    if not math.isfinite(radial_step) or radial_step <= 0.0:
        raise ValueError("论文风面源径向步长必须是正有限数。")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("论文风面源风速必须是非负有限数。")

    radius_km = float(selection.radius_km)
    ring_count = max(1, int(math.ceil(radius_km / radial_step)))
    edges_km = np.linspace(0.0, radius_km, ring_count + 1, dtype=np.float64)
    inner_km = edges_km[:-1]
    outer_km = edges_km[1:]
    centers_km = 0.5 * (inner_km + outer_km)
    bearings = np.arange(sectors, dtype=np.float64) * (360.0 / sectors)
    distance_grid, bearing_grid = np.meshgrid(centers_km, bearings, indexing="ij")
    area_per_sector = (
        math.pi
        * (np.square(outer_km * 1000.0) - np.square(inner_km * 1000.0))
        / sectors
    )
    area_grid = np.repeat(area_per_sector[:, None], sectors, axis=1)

    angular_distance = distance_grid * 1000.0 / EARTH_RADIUS_M
    bearing_rad = np.radians(bearing_grid)
    receiver_lat_rad = math.radians(float(selection.receiver_lat))
    receiver_lon_rad = math.radians(float(selection.receiver_lon))
    latitude_rad = np.arcsin(
        math.sin(receiver_lat_rad) * np.cos(angular_distance)
        + math.cos(receiver_lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    longitude_rad = receiver_lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * math.cos(receiver_lat_rad),
        np.cos(angular_distance)
        - math.sin(receiver_lat_rad) * np.sin(latitude_rad),
    )
    flat_distance = distance_grid.reshape(-1)
    return replace(
        selection,
        longitude=((np.degrees(longitude_rad).reshape(-1) + 180.0) % 360.0) - 180.0,
        latitude=np.degrees(latitude_rad).reshape(-1),
        distance_km=flat_distance,
        bearing_deg=bearing_grid.reshape(-1),
        wind_speed_mps=np.full(flat_distance.shape, speed, dtype=np.float64),
        cell_area_m2=area_grid.reshape(-1),
    )


def _utc_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("时间不能为空。")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _epoch_label(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(int(epoch_seconds), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _product_times(path: Path) -> np.ndarray:
    with Dataset(nc_path(path), "r") as dataset:
        if "time" not in dataset.variables:
            raise ValueError(f"风噪声源产品缺少 time：{path}")
        return np.asarray(dataset.variables["time"][:], dtype=np.int64).reshape(-1)


def resolve_source_product(
    product_paths: Iterable[Path],
    requested_time: Any = None,
    product_label: str = "风噪声",
) -> tuple[Path, int, str]:
    """从一个或多个可重复建模产品中解析请求时次。"""
    candidates: list[tuple[int, Path, int]] = []
    errors: list[str] = []
    for raw_path in product_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            for index, epoch in enumerate(_product_times(path)):
                candidates.append((int(epoch), path, index))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if not candidates:
        detail = f"；无法读取：{' | '.join(errors)}" if errors else ""
        raise FileNotFoundError(f"未找到可用的全球{product_label}源产品{detail}")

    candidates.sort(key=lambda item: (item[0], str(item[1])))
    if requested_time not in (None, ""):
        wanted = int(round(_utc_datetime(requested_time).timestamp()))
        matches = [item for item in candidates if item[0] == wanted]
        if not matches:
            available = ", ".join(_epoch_label(item[0]) for item in candidates[-8:])
            raise ValueError(
                f"没有与请求时间完全一致的{product_label}源产品。"
                f"可用时次：{available}"
            )
        epoch, path, index = matches[-1]
    else:
        epoch, path, index = candidates[-1]
    return path, index, _epoch_label(epoch)


def sample_wind_at_receiver(
    product_path: Path,
    time_index: int,
    receiver_lon: float,
    receiver_lat: float,
) -> ReceiverWindSample:
    """使用带掩膜的双线性权重，将 ERA5 u10/v10 插值到接收点。

    先插值矢量分量再计算风速，可避免直接插值风速大小所引入的偏差。
    """
    if not -90.0 <= float(receiver_lat) <= 90.0:
        raise ValueError("接收点纬度必须位于 -90 到 90°。")
    if not math.isfinite(float(receiver_lon)):
        raise ValueError("接收点经度必须是有限数值。")

    path = Path(product_path)
    with Dataset(nc_path(path), "r") as dataset:
        required = {"time", "latitude", "longitude", "u10", "v10", "source_valid_mask"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"风噪声源产品缺少论文基线模式所需变量：{', '.join(missing)}")
        time_values = np.asarray(dataset.variables["time"][:], dtype=np.int64).reshape(-1)
        if not 0 <= int(time_index) < time_values.size:
            raise IndexError("风噪声源产品的时间索引越界。")
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude = np.mod(np.asarray(dataset.variables["longitude"][:], dtype=np.float64), 360.0)
        u10 = np.asarray(
            np.ma.filled(dataset.variables["u10"][int(time_index), :, :], np.nan), dtype=np.float64
        )
        v10 = np.asarray(
            np.ma.filled(dataset.variables["v10"][int(time_index), :, :], np.nan), dtype=np.float64
        )
        valid = np.asarray(dataset.variables["source_valid_mask"][int(time_index), :, :], dtype=bool)
        time_utc = _epoch_label(int(time_values[int(time_index)]))

    if latitude.ndim != 1 or longitude.ndim != 1 or u10.shape != (latitude.size, longitude.size):
        raise ValueError("风噪声源产品的经纬度轴或 u10/v10 网格维度无效。")
    lat_order = np.argsort(latitude)
    lon_order = np.argsort(longitude)
    latitude = latitude[lat_order]
    longitude = longitude[lon_order]
    u10 = u10[np.ix_(lat_order, lon_order)]
    v10 = v10[np.ix_(lat_order, lon_order)]
    valid = valid[np.ix_(lat_order, lon_order)] & np.isfinite(u10) & np.isfinite(v10)

    target_lat = float(receiver_lat)
    target_lon = float(receiver_lon) % 360.0
    lat_upper = int(np.searchsorted(latitude, target_lat, side="right"))
    lat_upper = min(max(lat_upper, 1), latitude.size - 1)
    lat_lower = lat_upper - 1

    extended_lon = np.concatenate(([longitude[-1] - 360.0], longitude, [longitude[0] + 360.0]))
    extended_u = np.concatenate((u10[:, -1:], u10, u10[:, :1]), axis=1)
    extended_v = np.concatenate((v10[:, -1:], v10, v10[:, :1]), axis=1)
    extended_valid = np.concatenate((valid[:, -1:], valid, valid[:, :1]), axis=1)
    lon_upper = int(np.searchsorted(extended_lon, target_lon, side="right"))
    lon_upper = min(max(lon_upper, 1), extended_lon.size - 1)
    lon_lower = lon_upper - 1

    lat0, lat1 = float(latitude[lat_lower]), float(latitude[lat_upper])
    lon0, lon1 = float(extended_lon[lon_lower]), float(extended_lon[lon_upper])
    lat_fraction = 0.0 if lat1 == lat0 else float(np.clip((target_lat - lat0) / (lat1 - lat0), 0.0, 1.0))
    lon_fraction = 0.0 if lon1 == lon0 else float(np.clip((target_lon - lon0) / (lon1 - lon0), 0.0, 1.0))
    corners = [
        (lat_lower, lon_lower, (1.0 - lat_fraction) * (1.0 - lon_fraction)),
        (lat_lower, lon_upper, (1.0 - lat_fraction) * lon_fraction),
        (lat_upper, lon_lower, lat_fraction * (1.0 - lon_fraction)),
        (lat_upper, lon_upper, lat_fraction * lon_fraction),
    ]
    usable = [(i, j, weight) for i, j, weight in corners if extended_valid[i, j] and weight > 0.0]
    interpolation = "masked_bilinear_u10_v10"
    if usable:
        weight_sum = sum(item[2] for item in usable)
        u_value = sum(float(extended_u[i, j]) * weight for i, j, weight in usable) / weight_sum
        v_value = sum(float(extended_v[i, j]) * weight for i, j, weight in usable) / weight_sum
    else:
        # 非常靠近海岸的接收点可能四个角点都被掩膜。此时在局部搜索最近的有效
        # 海洋网格，而不使用陆地风场。
        centre_lat = int(np.argmin(np.abs(latitude - target_lat)))
        centre_lon = int(np.argmin(np.abs((longitude - target_lon + 180.0) % 360.0 - 180.0)))
        chosen = None
        for radius in range(1, 51):
            rows = np.arange(max(0, centre_lat - radius), min(latitude.size, centre_lat + radius + 1))
            columns = np.mod(np.arange(centre_lon - radius, centre_lon + radius + 1), longitude.size)
            candidates = [(int(i), int(j)) for i in rows for j in columns if valid[int(i), int(j)]]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda item: (
                        latitude[item[0]] - target_lat
                    ) ** 2 + (((longitude[item[1]] - target_lon + 180.0) % 360.0) - 180.0) ** 2,
                )
                break
        if chosen is None:
            raise ValueError("接收点附近没有可用的 ERA5 开阔海面风矢量。")
        u_value = float(u10[chosen])
        v_value = float(v10[chosen])
        interpolation = "nearest_valid_ocean_u10_v10"

    return ReceiverWindSample(
        product_path=path.resolve(),
        time_index=int(time_index),
        time_utc=time_utc,
        receiver_lon=((float(receiver_lon) + 180.0) % 360.0) - 180.0,
        receiver_lat=target_lat,
        u10_mps=u_value,
        v10_mps=v_value,
        wind_speed_mps=math.hypot(u_value, v_value),
        interpolation=interpolation,
    )


def _angular_distance_rad(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    receiver_lon_deg: float,
    receiver_lat_deg: float,
) -> np.ndarray:
    lon1 = np.deg2rad(float(receiver_lon_deg))
    lat1 = np.deg2rad(float(receiver_lat_deg))
    lon2 = np.deg2rad(longitude_deg)
    lat2 = np.deg2rad(latitude_deg)
    delta_lon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    delta_lat = lat2 - lat1
    hav = np.sin(delta_lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon * 0.5) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def _bearing_deg(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    receiver_lon_deg: float,
    receiver_lat_deg: float,
) -> np.ndarray:
    lon1 = math.radians(float(receiver_lon_deg))
    lat1 = math.radians(float(receiver_lat_deg))
    lon2 = np.deg2rad(longitude_deg)
    lat2 = np.deg2rad(latitude_deg)
    delta_lon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    east = np.sin(delta_lon) * np.cos(lat2)
    north = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return np.mod(np.rad2deg(np.arctan2(east, north)), 360.0)


def load_local_sources(
    product_path: Path,
    time_index: int,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    max_source_cells: int,
) -> WindSourceSelection:
    """只读取传播半径内的有效海面源，避免把全球四维数组整体载入内存。"""
    if not -90.0 <= float(receiver_lat) <= 90.0:
        raise ValueError("接收点纬度必须位于 -90 到 90°。")
    if not math.isfinite(float(receiver_lon)):
        raise ValueError("接收点经度必须是有限数值。")
    if not math.isfinite(float(radius_km)) or float(radius_km) <= 0.0:
        raise ValueError("风噪声传播半径必须为正数。")

    path = Path(product_path)
    with Dataset(nc_path(path), "r") as dataset:
        required = {
            "time", "latitude", "longitude", "source_valid_mask",
            "wind_speed_mps", "cell_area_m2",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"风噪声源产品缺少变量：{', '.join(missing)}")
        time_values = np.asarray(dataset.variables["time"][:], dtype=np.int64).reshape(-1)
        if not 0 <= int(time_index) < time_values.size:
            raise IndexError("风噪声源产品的时间索引越界。")
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude = np.mod(np.asarray(dataset.variables["longitude"][:], dtype=np.float64), 360.0)
        receiver_lon_360 = float(receiver_lon) % 360.0
        radius_angle = float(radius_km) * 1000.0 / EARTH_RADIUS_M
        latitude_margin = math.degrees(radius_angle) + 0.3
        latitude_indices = np.flatnonzero(np.abs(latitude - float(receiver_lat)) <= latitude_margin)
        cosine = max(abs(math.cos(math.radians(float(receiver_lat)))), 0.05)
        longitude_margin = min(180.0, math.degrees(radius_angle) / cosine + 0.3)
        longitude_delta = np.abs((longitude - receiver_lon_360 + 180.0) % 360.0 - 180.0)
        longitude_indices = np.flatnonzero(longitude_delta <= longitude_margin)
        if not latitude_indices.size or not longitude_indices.size:
            raise ValueError("风噪声源产品中没有覆盖接收点传播范围的网格。")

        lat_slice = slice(int(latitude_indices[0]), int(latitude_indices[-1]) + 1)
        # 经度索引在日期变更线附近可能不连续，使用显式索引保证正确。
        lat_values = latitude[latitude_indices]
        lon_values = longitude[longitude_indices]
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        valid = np.asarray(
            dataset.variables["source_valid_mask"][int(time_index), lat_slice, :][:, longitude_indices],
            dtype=bool,
        )
        wind = np.asarray(
            np.ma.filled(
                dataset.variables["wind_speed_mps"][int(time_index), lat_slice, :][:, longitude_indices],
                np.nan,
            ),
            dtype=np.float64,
        )
        area = np.asarray(
            np.ma.filled(dataset.variables["cell_area_m2"][lat_slice, :][:, longitude_indices], np.nan),
            dtype=np.float64,
        )
        source_depth = float(getattr(dataset, "source_depth_m", 1.0))
        reference_integral = float(getattr(dataset, "reference_transfer_integral_m2", math.pi))
        time_utc = _epoch_label(int(time_values[int(time_index)]))

    angular = _angular_distance_rad(lon_grid, lat_grid, receiver_lon_360, float(receiver_lat))
    distance_km = angular * EARTH_RADIUS_M / 1000.0
    keep = (
        valid
        & np.isfinite(wind)
        & (wind >= 0.0)
        & np.isfinite(area)
        & (area > 0.0)
        & (distance_km <= float(radius_km) + 1.0e-9)
    )
    if not np.any(keep):
        raise ValueError("接收点传播范围内没有有效的开阔海面风噪声源。")
    if int(np.count_nonzero(keep)) > int(max_source_cells):
        raise ValueError(
            f"传播范围内有 {int(np.count_nonzero(keep))} 个风噪声源格点，"
            f"超过服务上限 {int(max_source_cells)}；请减小 range。"
        )

    selected_area = area[keep]
    selected_distance = distance_km[keep]
    # 接收点所在网格代表一个面源，不能把格点中心重合误作零距离点源。
    equivalent_radius_km = 0.5 * np.sqrt(selected_area / math.pi) / 1000.0
    selected_distance = np.maximum(selected_distance, equivalent_radius_km)
    return WindSourceSelection(
        product_path=path.resolve(),
        time_index=int(time_index),
        time_utc=time_utc,
        receiver_lon=((float(receiver_lon) + 180.0) % 360.0) - 180.0,
        receiver_lat=float(receiver_lat),
        radius_km=float(radius_km),
        source_depth_m=source_depth,
        longitude=((lon_grid[keep] + 180.0) % 360.0) - 180.0,
        latitude=lat_grid[keep],
        distance_km=selected_distance,
        bearing_deg=_bearing_deg(lon_grid[keep], lat_grid[keep], receiver_lon_360, float(receiver_lat)),
        wind_speed_mps=wind[keep],
        cell_area_m2=selected_area,
        reference_transfer_integral_m2=reference_integral,
    )


def load_local_rain_sources(
    product_path: Path,
    time_index: int,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    max_source_cells: int,
) -> RainSourceSelection:
    """读取传播半径内的 ERA5 液态降水声源网格。"""
    path = Path(product_path)
    receiver_lon_360 = float(receiver_lon) % 360.0
    with Dataset(nc_path(path), "r") as dataset:
        required = {
            "time", "latitude", "longitude", "source_valid_mask",
            "rain_rate_mm_h", "cell_area_m2",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"降雨噪声源产品缺少变量：{', '.join(missing)}")
        time_values = np.asarray(dataset.variables["time"][:], dtype=np.int64).reshape(-1)
        if not 0 <= int(time_index) < time_values.size:
            raise IndexError("降雨噪声源产品的时间索引越界。")
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude = np.mod(np.asarray(dataset.variables["longitude"][:], dtype=np.float64), 360.0)
        radius_angle = float(radius_km) * 1000.0 / EARTH_RADIUS_M
        latitude_margin = math.degrees(radius_angle) + 0.3
        latitude_indices = np.flatnonzero(np.abs(latitude - float(receiver_lat)) <= latitude_margin)
        cosine = max(abs(math.cos(math.radians(float(receiver_lat)))), 0.05)
        longitude_margin = min(180.0, math.degrees(radius_angle) / cosine + 0.3)
        longitude_delta = np.abs((longitude - receiver_lon_360 + 180.0) % 360.0 - 180.0)
        longitude_indices = np.flatnonzero(longitude_delta <= longitude_margin)
        if not latitude_indices.size or not longitude_indices.size:
            raise ValueError("降雨噪声源产品中没有覆盖接收点传播范围的网格。")
        lat_slice = slice(int(latitude_indices[0]), int(latitude_indices[-1]) + 1)
        lat_values = latitude[latitude_indices]
        lon_values = longitude[longitude_indices]
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        valid = np.asarray(
            dataset.variables["source_valid_mask"][int(time_index), lat_slice, :][:, longitude_indices],
            dtype=bool,
        )
        rain_rate = np.asarray(
            np.ma.filled(
                dataset.variables["rain_rate_mm_h"][int(time_index), lat_slice, :][:, longitude_indices],
                np.nan,
            ),
            dtype=np.float64,
        )
        area = np.asarray(
            np.ma.filled(dataset.variables["cell_area_m2"][lat_slice, :][:, longitude_indices], np.nan),
            dtype=np.float64,
        )
        source_depth = float(getattr(dataset, "source_depth_m", 1.0))
        reference_integral = float(getattr(dataset, "reference_transfer_integral_m2", math.pi))
        time_utc = _epoch_label(int(time_values[int(time_index)]))

    angular = _angular_distance_rad(lon_grid, lat_grid, receiver_lon_360, float(receiver_lat))
    distance_km = angular * EARTH_RADIUS_M / 1000.0
    keep = (
        valid
        & np.isfinite(rain_rate)
        & (rain_rate > 0.0)
        & np.isfinite(area)
        & (area > 0.0)
        & (distance_km <= float(radius_km) + 1.0e-9)
    )
    count = int(np.count_nonzero(keep))
    if count > int(max_source_cells):
        raise ValueError(
            f"传播范围内有 {count} 个降雨噪声源格点，"
            f"超过服务上限 {int(max_source_cells)}；请减小 range。"
        )
    selected_area = area[keep]
    selected_distance = distance_km[keep]
    if count:
        equivalent_radius_km = 0.5 * np.sqrt(selected_area / math.pi) / 1000.0
        selected_distance = np.maximum(selected_distance, equivalent_radius_km)
    return RainSourceSelection(
        product_path=path.resolve(),
        time_index=int(time_index),
        time_utc=time_utc,
        receiver_lon=((float(receiver_lon) + 180.0) % 360.0) - 180.0,
        receiver_lat=float(receiver_lat),
        radius_km=float(radius_km),
        source_depth_m=source_depth,
        longitude=((lon_grid[keep] + 180.0) % 360.0) - 180.0,
        latitude=lat_grid[keep],
        distance_km=selected_distance,
        bearing_deg=_bearing_deg(lon_grid[keep], lat_grid[keep], receiver_lon_360, float(receiver_lat)),
        rain_rate_mm_h=rain_rate[keep],
        cell_area_m2=selected_area,
        reference_transfer_integral_m2=reference_integral,
    )


def load_local_shipping_sources(
    product_path: Path,
    time_index: int,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    max_source_cells: int,
) -> ShippingSourceSelection:
    """读取传播半径内的 AIS/GFW 稀疏航运声源网格。"""
    path = Path(product_path)
    with Dataset(nc_path(path), "r") as dataset:
        required = {
            "time", "lon", "lat", "frequency_hz", "source_depth_m",
            "cell_area_m2", "cell_source_power_linear",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"AIS 航运噪声源产品缺少变量：{', '.join(missing)}")
        time_values = np.asarray(dataset.variables["time"][:], dtype=np.int64).reshape(-1)
        if not 0 <= int(time_index) < time_values.size:
            raise IndexError("AIS 航运噪声源产品的时间索引越界。")
        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["cell_area_m2"][:], dtype=np.float64)
        source_depth_values = np.asarray(dataset.variables["source_depth_m"][:], dtype=np.float64)
        frequency = np.asarray(dataset.variables["frequency_hz"][:], dtype=np.float64)
        power = np.asarray(
            np.ma.filled(dataset.variables["cell_source_power_linear"][:], np.nan),
            dtype=np.float64,
        )
        time_utc = _epoch_label(int(time_values[int(time_index)]))
    if power.shape != (frequency.size, longitude.size):
        raise ValueError("AIS 航运噪声源产品的功率谱维度无效。")
    angular = _angular_distance_rad(longitude, latitude, float(receiver_lon), float(receiver_lat))
    distance_km = angular * EARTH_RADIUS_M / 1000.0
    active = np.any(np.isfinite(power) & (power > 0.0), axis=0)
    keep = (
        active
        & np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(area)
        & (area > 0.0)
        & (distance_km <= float(radius_km) + 1.0e-9)
    )
    count = int(np.count_nonzero(keep))
    if count > int(max_source_cells):
        raise ValueError(
            f"传播范围内有 {count} 个 AIS 航运噪声源格点，"
            f"超过服务上限 {int(max_source_cells)}；请减小 range。"
        )
    selected_distance = distance_km[keep]
    if count:
        equivalent_radius_km = 0.5 * np.sqrt(area[keep] / math.pi) / 1000.0
        selected_distance = np.maximum(selected_distance, equivalent_radius_km)
    valid_depths = source_depth_values[np.isfinite(source_depth_values) & (source_depth_values > 0.0)]
    source_depth = float(np.median(valid_depths)) if valid_depths.size else 5.0
    return ShippingSourceSelection(
        product_path=path.resolve(),
        time_index=int(time_index),
        time_utc=time_utc,
        receiver_lon=((float(receiver_lon) + 180.0) % 360.0) - 180.0,
        receiver_lat=float(receiver_lat),
        radius_km=float(radius_km),
        source_depth_m=source_depth,
        longitude=((longitude[keep] + 180.0) % 360.0) - 180.0,
        latitude=latitude[keep],
        distance_km=selected_distance,
        bearing_deg=_bearing_deg(longitude[keep], latitude[keep], receiver_lon, receiver_lat),
        frequency_hz=frequency,
        cell_source_power_linear=power[:, keep],
    )


def propagation_frequencies(minimum_hz: float, maximum_hz: float, count: int) -> np.ndarray:
    """返回用于实际声传播计算的对数均匀频率锚点。"""
    minimum = float(minimum_hz)
    maximum = float(maximum_hz)
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("传播频率范围必须满足 0 < minFreq < maxFreq。")
    return np.geomspace(minimum, maximum, max(2, int(count)), dtype=np.float64)


def wind_cell_source_psd_linear(selection: WindSourceSelection, frequency_hz: float) -> np.ndarray:
    frequency_khz = float(frequency_hz) / 1000.0
    if not math.isfinite(frequency_khz) or frequency_khz <= 0.0:
        raise ValueError("风噪声频率必须为正数。")
    reference_db = (
        50.0
        + 7.5 * np.sqrt(selection.wind_speed_mps)
        + 20.0 * np.log10(frequency_khz)
        - 40.0 * np.log10(frequency_khz + 0.4)
    )
    density = np.power(10.0, reference_db / 10.0) / selection.reference_transfer_integral_m2
    return density * selection.cell_area_m2


def liu2022_wind_surface_source_level_db(
    frequency_hz: float,
    wind_speed_mps: Any,
) -> np.ndarray:
    """东印度洋论文式(14)的单位面积风生噪声源级。

    该区域拟合式仅在 500–6400 Hz 有效：
    ``SLW=48.78-3.74log10((f/400)^2+1)``
    ``+(11.85+U/2.06)log10(U/5.15)``。
    """
    frequency = float(frequency_hz)
    if not RAIN_MODEL_MIN_FREQUENCY_HZ <= frequency <= RAIN_MODEL_MAX_FREQUENCY_HZ:
        raise ValueError("论文风噪声面源模型频率范围为 500–6400 Hz。")
    speed = np.asarray(wind_speed_mps, dtype=np.float64)
    if np.any(~np.isfinite(speed)) or np.any(speed < 0.0):
        raise ValueError("风速必须是非负有限数，单位为 m/s。")
    level = np.full(speed.shape, -np.inf, dtype=np.float64)
    active = speed > 0.0
    level[active] = (
        48.78
        - 3.74 * math.log10((frequency / 400.0) ** 2 + 1.0)
        + (11.85 + speed[active] / 2.06) * np.log10(speed[active] / 5.15)
    )
    return level


def liu2022_wind_cell_source_power_linear(
    selection: WindSourceSelection,
    frequency_hz: float,
) -> np.ndarray:
    """把论文单位面积风源级乘以各 ERA5 海面网格面积。"""
    if selection.source_count == 0:
        return np.empty(0, dtype=np.float64)
    source_level_db = liu2022_wind_surface_source_level_db(
        frequency_hz,
        selection.wind_speed_mps,
    )
    density = np.zeros(source_level_db.shape, dtype=np.float64)
    active = np.isfinite(source_level_db)
    density[active] = np.power(10.0, source_level_db[active] / 10.0)
    return density * selection.cell_area_m2


def rain_cell_source_psd_linear(selection: RainSourceSelection, frequency_hz: float) -> np.ndarray:
    """按论文式(8)计算每个非均匀降雨网格的声源功率。

    单位面积源级 ``SLR=51+10log10(R)`` 先转为线性强度，再乘以
    球面网格的实际面积，对应论文式(5)中的 ``ns^2 * S``。
    """
    if selection.source_count == 0:
        return np.empty(0, dtype=np.float64)
    frequency = float(frequency_hz)
    if not RAIN_MODEL_MIN_FREQUENCY_HZ <= frequency <= RAIN_MODEL_MAX_FREQUENCY_HZ:
        raise ValueError("论文降雨面源模型频率范围为 500–6400 Hz。")
    from .environment_noise import rain_surface_source_level_db

    source_level_db = rain_surface_source_level_db(selection.rain_rate_mm_h)
    density = np.power(10.0, source_level_db / 10.0)
    return density * selection.cell_area_m2


def rain_surface_source_depth_m(frequency_hz: float) -> float:
    """论文模型中的海面下 1/4 波长噪声源深度。"""
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("降雨噪声频率必须是正有限数。")
    return RAIN_SOURCE_REFERENCE_SOUND_SPEED_MPS / (4.0 * frequency)


def shipping_cell_source_power_linear(
    selection: ShippingSourceSelection,
    frequency_hz: float,
) -> np.ndarray:
    """AIS 产品的参考频点在对数频率/分贝域插值。

    产品只有 63 Hz 和 1000 Hz 文献参考值；范围外使用最近参考频点，
    避免将两点谱型不受控地外推。
    """
    if selection.source_count == 0:
        return np.empty(0, dtype=np.float64)
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("航运噪声频率必须为正数。")
    order = np.argsort(selection.frequency_hz)
    reference_frequency = np.asarray(selection.frequency_hz[order], dtype=np.float64)
    reference_power = np.asarray(selection.cell_source_power_linear[order], dtype=np.float64)
    reference_db = 10.0 * np.log10(np.maximum(reference_power, 1.0e-300))
    if frequency <= reference_frequency[0]:
        level_db = reference_db[0]
    elif frequency >= reference_frequency[-1]:
        level_db = reference_db[-1]
    else:
        upper = int(np.searchsorted(reference_frequency, frequency, side="right"))
        lower = upper - 1
        weight = (
            (math.log(frequency) - math.log(float(reference_frequency[lower])))
            / (math.log(float(reference_frequency[upper])) - math.log(float(reference_frequency[lower])))
        )
        level_db = reference_db[lower] + (reference_db[upper] - reference_db[lower]) * weight
    return np.power(10.0, level_db / 10.0)


def _depth_layer(dataset: Dataset, source_depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(dataset.variables["depth_m"][:], dtype=np.float64)
    theta = np.mod(np.asarray(dataset.variables["theta_deg"][:], dtype=np.float64), 360.0)
    ranges = np.asarray(dataset.variables["range_km"][:], dtype=np.float64)
    tl = np.asarray(np.ma.filled(dataset.variables["tl_db"][0], np.nan), dtype=np.float64)
    if tl.shape != (depth.size, theta.size, ranges.size):
        raise ValueError("传播损失文件的 tl_db 维度与 depth/theta/range 不一致。")
    if source_depth_m < float(np.min(depth)) - 1.0e-6 or source_depth_m > float(np.max(depth)) + 1.0e-6:
        raise ValueError(f"传播损失深度网格不包含海面源深度 {source_depth_m:g} m。")
    upper = int(np.searchsorted(depth, source_depth_m, side="left"))
    if upper <= 0:
        layer = tl[0]
    elif upper >= depth.size:
        layer = tl[-1]
    elif abs(float(depth[upper]) - source_depth_m) <= 1.0e-6:
        layer = tl[upper]
    else:
        lower = upper - 1
        weight = (source_depth_m - depth[lower]) / (depth[upper] - depth[lower])
        layer = tl[lower] + (tl[upper] - tl[lower]) * weight
    return theta, ranges, layer


def sample_source_transmission_loss(
    selection: WindSourceSelection,
    tl_path: Path,
    *,
    source_depth_m: float | None = None,
) -> np.ndarray:
    """在各海面源方位/距离采样互易传播场。方位采用最近扇区，距离线性插值。"""
    with Dataset(nc_path(tl_path), "r") as dataset:
        required = {"depth_m", "theta_deg", "range_km", "tl_db"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"传播损失文件缺少变量：{', '.join(missing)}")
        depth_m = selection.source_depth_m if source_depth_m is None else float(source_depth_m)
        theta, ranges, layer = _depth_layer(dataset, depth_m)

    if theta.size < 1 or ranges.size < 2 or np.any(np.diff(ranges) <= 0.0):
        raise ValueError("传播损失文件的方位或距离轴无效。")
    angle_delta = np.abs((theta[:, None] - selection.bearing_deg[None, :] + 180.0) % 360.0 - 180.0)
    theta_index = np.argmin(angle_delta, axis=0)
    result = np.full(selection.source_count, np.nan, dtype=np.float64)
    for index in range(selection.source_count):
        values = np.asarray(layer[int(theta_index[index])], dtype=np.float64)
        good = np.isfinite(values) & np.isfinite(ranges)
        if np.count_nonzero(good) < 2:
            continue
        x = ranges[good]
        y = values[good]
        order = np.argsort(x)
        distance = float(selection.distance_km[index])
        if distance < float(x[order][0]) - 1.0e-7 or distance > float(x[order][-1]) + 1.0e-7:
            continue
        result[index] = float(np.interp(distance, x[order], y[order]))
    return result


def propagated_wind_level_db(
    selection: WindSourceSelection,
    frequency_hz: float,
    tl_path: Path,
) -> tuple[float, int]:
    """以线性功率叠加全部可传播海面源，并返回接收点风噪声谱级。"""
    transmission_loss = sample_source_transmission_loss(selection, tl_path)
    source_power = wind_cell_source_psd_linear(selection, frequency_hz)
    valid = np.isfinite(transmission_loss) & np.isfinite(source_power) & (source_power > 0.0)
    if not np.any(valid):
        raise ValueError("传播模型没有给出任何可到达接收点的有效风噪声源。")
    receiver_power = np.sum(source_power[valid] * np.power(10.0, -transmission_loss[valid] / 10.0))
    if not math.isfinite(float(receiver_power)) or float(receiver_power) <= 0.0:
        raise ValueError("风噪声传播功率叠加结果无效。")
    return float(10.0 * math.log10(float(receiver_power))), int(np.count_nonzero(valid))


def _propagated_component_level_db(
    selection: Any,
    source_power: np.ndarray,
    tl_path: Path,
    label: str,
    *,
    source_depth_m: float | None = None,
) -> tuple[float, int]:
    if selection.source_count == 0:
        return -300.0, 0
    transmission_loss = sample_source_transmission_loss(
        selection,
        tl_path,
        source_depth_m=source_depth_m,
    )
    power = np.asarray(source_power, dtype=np.float64)
    valid = np.isfinite(transmission_loss) & np.isfinite(power) & (power > 0.0)
    if not np.any(valid):
        raise ValueError(f"传播模型没有给出任何可到达接收点的有效{label}源。")
    receiver_power = np.sum(power[valid] * np.power(10.0, -transmission_loss[valid] / 10.0))
    if not math.isfinite(float(receiver_power)) or float(receiver_power) <= 0.0:
        raise ValueError(f"{label}传播功率叠加结果无效。")
    return float(10.0 * math.log10(float(receiver_power))), int(np.count_nonzero(valid))


def propagated_rain_level_db(
    selection: RainSourceSelection,
    frequency_hz: float,
    tl_path: Path,
) -> tuple[float, int]:
    return _propagated_component_level_db(
        selection,
        rain_cell_source_psd_linear(selection, frequency_hz),
        tl_path,
        "降雨噪声",
        source_depth_m=rain_surface_source_depth_m(frequency_hz),
    )


def propagated_liu2022_wind_level_db(
    selection: WindSourceSelection,
    frequency_hz: float,
    tl_path: Path,
) -> tuple[float, int]:
    return _propagated_component_level_db(
        selection,
        liu2022_wind_cell_source_power_linear(selection, frequency_hz),
        tl_path,
        "论文风噪声",
        source_depth_m=rain_surface_source_depth_m(frequency_hz),
    )


def propagated_shipping_level_db(
    selection: ShippingSourceSelection,
    frequency_hz: float,
    tl_path: Path,
) -> tuple[float, int]:
    return _propagated_component_level_db(
        selection,
        shipping_cell_source_power_linear(selection, frequency_hz),
        tl_path,
        "AIS 航运噪声",
    )


def distributed_response(
    params: Mapping[str, Any],
    selections: Mapping[str, Any],
    propagated_levels_db: Mapping[str, Mapping[float, float]],
    job_ids: Iterable[str],
) -> dict[str, Any]:
    """把混合模型分量求值到接口频率轴并在线性功率域合成。

    风分量可以是直接求值的 Hildebrand 接收谱，也可以是论文式(14)经传播
    得到的非均匀面源；雨和航运分量来自传播锚点。
    雨噪声在论文验证的 500–6400 Hz 之外以 ``null`` 返回且不参与总谱，
    避免把无效频段误写成零噪声或继续外推。
    """

    current_20hz = bool(params.get("_current20HzModel"))
    frequency = np.linspace(
        float(params["minFreq"]), float(params["maxFreq"]), int(params["numPoints"]), dtype=np.float64
    )
    components = {}
    component_output = {}
    frequency_anchors = {}
    for name in ("wind", "rain", "shipping"):
        source_levels = propagated_levels_db.get(name) or {}
        anchors = np.asarray(sorted(float(value) for value in source_levels), dtype=np.float64)
        levels = np.asarray([float(source_levels[float(value)]) for value in anchors], dtype=np.float64)
        if anchors.size == 0:
            values = np.full(frequency.shape, -np.inf, dtype=np.float64)
        elif anchors.size == 1:
            values = np.full(frequency.shape, levels[0], dtype=np.float64)
        elif np.any(~np.isfinite(levels)):
            raise ValueError(f"{name}噪声频率锚点包含无效谱级。")
        else:
            values = np.interp(np.log(frequency), np.log(anchors), levels)

        available = np.ones(frequency.shape, dtype=bool)
        if name == "wind" and params.get("_windSurfaceModel"):
            available &= (
                (frequency >= RAIN_MODEL_MIN_FREQUENCY_HZ)
                & (frequency <= RAIN_MODEL_MAX_FREQUENCY_HZ)
            )
        if name == "rain":
            rain_minimum = 500.0 if current_20hz else RAIN_MODEL_MIN_FREQUENCY_HZ
            rain_maximum = 1000.0 if current_20hz else RAIN_MODEL_MAX_FREQUENCY_HZ
            available &= (frequency >= rain_minimum) & (frequency <= rain_maximum)
        values = np.where(available, values, -np.inf)
        components[name] = values
        component_output[name] = [
            round(float(value), 6) if math.isfinite(float(value)) else None
            for value in values
        ]
        # Hildebrand 是逐响应频点直接求值，不是传播锚点。
        frequency_anchors[name] = (
            [] if name == "wind" and params.get("_windModel") else np.round(anchors, 6).tolist()
        )

    component_power = np.zeros(frequency.shape, dtype=np.float64)
    for values in components.values():
        finite = np.isfinite(values)
        component_power[finite] += np.power(10.0, values[finite] / 10.0)
    if np.any(component_power <= 0.0) or np.any(~np.isfinite(component_power)):
        raise ValueError("混合环境噪声没有可用于合成的有效分量。")
    total_db = 10.0 * np.log10(component_power)
    first_selection = next(iter(selections.values()))
    source_models = {
        name: {
            "time": selection.time_utc,
            "sourceDepthM": round(selection.source_depth_m, 6),
            "radiusKm": round(selection.radius_km, 6),
            "sourceCellCount": selection.source_count,
            "product": selection.product_path.name,
            "calibration": "equivalent_source_uncalibrated",
        }
        for name, selection in selections.items()
    }
    if params.get("_windModel"):
        source_models["wind"] = dict(params["_windModel"])
        source_models["wind"]["calibration"] = "empirical_received_psd"
        source_models["wind"]["propagatedAgain"] = False
    elif params.get("_windSurfaceModel"):
        source_models["wind"].pop("sourceDepthM", None)
        source_models["wind"].update(dict(params["_windSurfaceModel"]))
    if current_20hz:
        for name, metadata in dict(params.get("_current20HzSourceModels") or {}).items():
            source_models[name].update(dict(metadata))
        source_models["wind"].pop("sourceDepthM", None)
        source_models["rain"].pop("sourceDepthM", None)
        source_models["wind"].update({
            "sourceDepthRule": "max(0.25,c/(4f))",
            "sourceDepthRangeM": [0.375, 18.75],
            "referenceSoundSpeedMps": RAIN_SOURCE_REFERENCE_SOUND_SPEED_MPS,
        })
        source_models["rain"].update({
            "sourceDepthRule": "max(0.25,c/(4f))",
            "sourceDepthRangeM": [0.375, 0.75],
            "referenceSoundSpeedMps": RAIN_SOURCE_REFERENCE_SOUND_SPEED_MPS,
        })
    else:
        source_models["rain"].pop("sourceDepthM", None)
        source_models["rain"].update({
            "sourceDepthRule": "c/(4f)",
            "sourceDepthRangeM": [
                round(rain_surface_source_depth_m(RAIN_MODEL_MAX_FREQUENCY_HZ), 6),
                round(rain_surface_source_depth_m(RAIN_MODEL_MIN_FREQUENCY_HZ), 6),
            ],
            "referenceSoundSpeedMps": RAIN_SOURCE_REFERENCE_SOUND_SPEED_MPS,
            "calibration": "published_unit_area_source_level",
            "model": "Liu2022_Urick_SLR",
            "equation": "SLR=51+10log10(R_mm_h)",
            "reference": "Liu Yunfeng et al. (2022), equations (5) and (8)",
        })
    return {
        "freq": np.round(frequency, 6).tolist(),
        "data": np.round(total_db, 6).tolist(),
        "mode": "distributed_noise_source_grid",
        "sourceGridModel": (
            dict(params["_current20HzModel"])
            if current_20hz
            else {"id": "legacy"}
        ),
        "receiver": {
            "longitude": round(first_selection.receiver_lon, 6),
            "latitude": round(first_selection.receiver_lat, 6),
            "depth_m": round(float(params["receiverDepthM"]), 6),
        },
        "sourceModels": source_models,
        "propagation": {
            "model": "hybrid_by_frequency",
            "reciprocity": True,
            "frequencyInterpolation": "linear_level_on_log_frequency",
            "frequencyAnchorsHz": frequency_anchors,
            "jobIds": list(job_ids),
            **dict(params.get("_hybridPropagation") or {}),
        },
        "componentValidity": ({
            "wind": {
                "frequencyHz": [20.0, 1000.0],
                "model": "Ainslie 2010 areic point-dipole source spectrum",
            },
            "rain": {
                "frequencyHz": [500.0, 1000.0],
                "model": "Urick amplitude plus Ma-Nystuen-inspired shape",
            },
            "shipping": {
                "frequencyHz": [20.0, 1000.0],
                "model": "JOMOPANS-ECHO AIS source spectrum",
            },
        } if current_20hz else {
            "wind": (
                {
                    "frequencyHz": [RAIN_MODEL_MIN_FREQUENCY_HZ, RAIN_MODEL_MAX_FREQUENCY_HZ],
                    "model": "Liu2022 East Indian Ocean nonuniform unit-area surface sources",
                }
                if params.get("_windSurfaceModel")
                else {"frequencyHz": [10.0, 20000.0], "model": "Hildebrand2021 received PSD"}
            ),
            "rain": {
                "frequencyHz": [RAIN_MODEL_MIN_FREQUENCY_HZ, RAIN_MODEL_MAX_FREQUENCY_HZ],
                "model": "Liu2022/Urick nonuniform unit-area surface sources",
            },
            "shipping": {"referenceFrequencyHz": [63.0, 1000.0], "model": "GFW class proxy"},
        }),
        "components": component_output,
    }
