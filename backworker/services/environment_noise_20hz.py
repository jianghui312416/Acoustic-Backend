"""20 Hz 分辨率的风、雨和 AIS 船舶分布式声源产品。

本模块只负责读取已经生成的源网格、截取接收点周围的源单元，以及把
各单元的源谱功率与 Bellhop 传播损失在线性功率域相加。传播任务仍由
门户接口的统一声场作业系统负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np

from .wind_noise_receiver import nc_path, sample_source_transmission_loss


MODEL_ID = "ainslie_urick_jomopans_20hz"
MODEL_TIME_UTC = "2025-01-02T12:00:00Z"
MIN_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 1000.0
FREQUENCY_STEP_HZ = 20.0
RAIN_MIN_FREQUENCY_HZ = 500.0
RAIN_MAX_FREQUENCY_HZ = 1000.0
REFERENCE_SOUND_SPEED_MPS = 1500.0
MIN_SURFACE_SOURCE_DEPTH_M = 0.25
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class SpectralSourceSelection:
    """接收点周围、带完整 20 Hz 源谱的稀疏声源集合。"""

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
    model: str

    @property
    def source_count(self) -> int:
        return int(self.distance_km.size)


def _normalise_utc(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("20 Hz 环境噪声模型必须传入 time。")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"无法解析请求时间 {value!r}，应使用 yyyy-MM-dd HH:mm:ss。") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_model_time(requested_time: Any, model_time_utc: str = MODEL_TIME_UTC) -> str:
    requested = _normalise_utc(requested_time)
    expected = _normalise_utc(model_time_utc)
    if requested != expected:
        raise ValueError(
            f"当前 20 Hz 源网格仅对应 {expected}；收到的请求时次为 {requested}。"
        )
    return expected


def native_frequency_axis(minimum_hz: float, maximum_hz: float) -> np.ndarray:
    minimum = float(minimum_hz)
    maximum = float(maximum_hz)
    if minimum < MIN_FREQUENCY_HZ or maximum > MAX_FREQUENCY_HZ:
        raise ValueError("当前 20 Hz 模型频率范围为 20–1000 Hz。")
    first_index = round((minimum - MIN_FREQUENCY_HZ) / FREQUENCY_STEP_HZ)
    last_index = round((maximum - MIN_FREQUENCY_HZ) / FREQUENCY_STEP_HZ)
    first = MIN_FREQUENCY_HZ + first_index * FREQUENCY_STEP_HZ
    last = MIN_FREQUENCY_HZ + last_index * FREQUENCY_STEP_HZ
    if not math.isclose(first, minimum, abs_tol=1.0e-7) or not math.isclose(last, maximum, abs_tol=1.0e-7):
        raise ValueError("当前 20 Hz 模型的 minFreq 和 maxFreq 必须落在 20 Hz 频率网格上。")
    return np.arange(first, last + 0.5 * FREQUENCY_STEP_HZ, FREQUENCY_STEP_HZ, dtype=np.float64)


def surface_source_depth_m(frequency_hz: float, water_depth_m: float | None = None) -> float:
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("声源频率必须是正有限数。")
    depth = max(MIN_SURFACE_SOURCE_DEPTH_M, REFERENCE_SOUND_SPEED_MPS / (4.0 * frequency))
    if water_depth_m is not None:
        water_depth = float(water_depth_m)
        if not math.isfinite(water_depth) or water_depth <= 0.0:
            raise ValueError("海水深度必须是正有限数。")
        depth = min(depth, max(MIN_SURFACE_SOURCE_DEPTH_M, 0.25 * water_depth))
    return depth


def shipping_source_depth_m(nominal_depth_m: float, water_depth_m: float) -> float:
    water_depth = float(water_depth_m)
    nominal_depth = float(nominal_depth_m)
    if not math.isfinite(water_depth) or water_depth <= 0.0:
        raise ValueError("海水深度必须是正有限数。")
    if not math.isfinite(nominal_depth) or nominal_depth <= 0.0:
        raise ValueError("船舶声源深度必须是正有限数。")
    return min(nominal_depth, max(0.5, 0.75 * water_depth))


def _wrap_lon_delta(longitude: np.ndarray, centre_lon: float) -> np.ndarray:
    return (np.asarray(longitude, dtype=np.float64) - float(centre_lon) + 180.0) % 360.0 - 180.0


def _distance_km(longitude: np.ndarray, latitude: np.ndarray, centre_lon: float, centre_lat: float) -> np.ndarray:
    lat1 = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    lat2 = math.radians(float(centre_lat))
    dlat = lat1 - lat2
    dlon = np.deg2rad(_wrap_lon_delta(longitude, centre_lon))
    value = np.sin(dlat / 2.0) ** 2 + math.cos(lat2) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def _bearing_deg(longitude: np.ndarray, latitude: np.ndarray, centre_lon: float, centre_lat: float) -> np.ndarray:
    lat1 = math.radians(float(centre_lat))
    lat2 = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    dlon = np.deg2rad(_wrap_lon_delta(longitude, centre_lon))
    y = np.sin(dlon) * np.cos(lat2)
    x = math.cos(lat1) * np.sin(lat2) - math.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def _limit_nearest(indices: np.ndarray, distance: np.ndarray, maximum_cells: int) -> np.ndarray:
    maximum = int(maximum_cells)
    if maximum < 1:
        raise ValueError("最大声源网格数必须大于 0。")
    if indices.size <= maximum:
        return indices
    return indices[np.argsort(distance[indices], kind="stable")[:maximum]]


def _paired_spectrum(variable: Any, lat_index: np.ndarray, lon_index: np.ndarray) -> np.ndarray:
    values = np.empty((variable.shape[1], lat_index.size), dtype=np.float64)
    for column, (i, j) in enumerate(zip(lat_index, lon_index)):
        values[:, column] = np.asarray(
            np.ma.filled(variable[0, :, int(i), int(j)], 0.0), dtype=np.float64
        )
    return values


def _read_regular_components(
    path: Path,
    model_time_utc: str,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    maximum_cells: int,
) -> tuple[SpectralSourceSelection, SpectralSourceSelection]:
    with Dataset(nc_path(path), "r") as dataset:
        required = {
            "frequency_hz", "latitude", "longitude", "cell_area_m2", "ocean_mask",
            "wind_cell_source_factor_linear", "rain_cell_source_factor_linear",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"20 Hz 风雨源产品缺少变量：{', '.join(missing)}")
        frequency = np.asarray(dataset.variables["frequency_hz"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
        area = np.asarray(np.ma.filled(dataset.variables["cell_area_m2"][:], np.nan), dtype=np.float64)
        ocean = np.asarray(dataset.variables["ocean_mask"][:], dtype=bool)

        lat_margin = radius_km / 110.574 + 0.3
        lon_margin = radius_km / max(20.0, 111.320 * abs(math.cos(math.radians(receiver_lat)))) + 0.3
        lat_candidates = np.flatnonzero(np.abs(latitude - receiver_lat) <= lat_margin)
        lon_candidates = np.flatnonzero(np.abs(_wrap_lon_delta(longitude, receiver_lon)) <= lon_margin)
        yy, xx = np.meshgrid(lat_candidates, lon_candidates, indexing="ij")
        flat_lat = yy.ravel()
        flat_lon = xx.ravel()
        candidate_distance = _distance_km(
            longitude[flat_lon], latitude[flat_lat], receiver_lon, receiver_lat
        )
        keep = (
            (candidate_distance <= radius_km + 1.0e-9)
            & ocean[flat_lat, flat_lon]
            & np.isfinite(area[flat_lat, flat_lon])
            & (area[flat_lat, flat_lon] > 0.0)
        )
        selected = np.flatnonzero(keep)
        selected = _limit_nearest(selected, candidate_distance, maximum_cells)
        lat_index = flat_lat[selected].astype(np.int64)
        lon_index = flat_lon[selected].astype(np.int64)
        selected_area = area[lat_index, lon_index]
        distance = candidate_distance[selected]
        equivalent_radius = np.sqrt(selected_area / math.pi) / 2000.0
        distance = np.maximum(distance, equivalent_radius)
        selected_lon = longitude[lon_index]
        selected_lat = latitude[lat_index]
        wind_power = _paired_spectrum(
            dataset.variables["wind_cell_source_factor_linear"], lat_index, lon_index
        )
        rain_power = _paired_spectrum(
            dataset.variables["rain_cell_source_factor_linear"], lat_index, lon_index
        )

    common = dict(
        product_path=path,
        time_index=0,
        time_utc=model_time_utc,
        receiver_lon=float(receiver_lon),
        receiver_lat=float(receiver_lat),
        radius_km=float(radius_km),
        source_depth_m=MIN_SURFACE_SOURCE_DEPTH_M,
        longitude=selected_lon,
        latitude=selected_lat,
        distance_km=distance,
        bearing_deg=_bearing_deg(selected_lon, selected_lat, receiver_lon, receiver_lat),
        frequency_hz=frequency,
    )
    return (
        SpectralSourceSelection(
            **common,
            cell_source_power_linear=wind_power,
            model="Ainslie_2010_areic_point_dipole",
        ),
        SpectralSourceSelection(
            **common,
            cell_source_power_linear=rain_power,
            model="Urick_amplitude_Ma_Nystuen_inspired_shape",
        ),
    )


def _read_shipping(
    path: Path,
    model_time_utc: str,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    maximum_cells: int,
) -> SpectralSourceSelection:
    with Dataset(nc_path(path), "r") as dataset:
        required = {
            "frequency_hz", "lon", "lat", "cell_area_m2", "source_depth_m",
            "cell_source_spectral_density_linear",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"20 Hz AIS 船舶源产品缺少变量：{', '.join(missing)}")
        frequency = np.asarray(dataset.variables["frequency_hz"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        area = np.asarray(np.ma.filled(dataset.variables["cell_area_m2"][:], np.nan), dtype=np.float64)
        depth_values = np.asarray(dataset.variables["source_depth_m"][:], dtype=np.float64)
        distance = _distance_km(longitude, latitude, receiver_lon, receiver_lat)
        keep = (
            np.isfinite(distance)
            & (distance <= radius_km + 1.0e-9)
            & np.isfinite(area)
            & (area > 0.0)
        )
        selected = _limit_nearest(np.flatnonzero(keep), distance, maximum_cells)
        selected_area = area[selected]
        equivalent_radius = np.sqrt(selected_area / math.pi) / 2000.0
        selected_distance = np.maximum(distance[selected], equivalent_radius)
        if selected.size:
            power = np.asarray(
                np.ma.filled(
                    dataset.variables["cell_source_spectral_density_linear"][:, selected],
                    0.0,
                ),
                dtype=np.float64,
            ).reshape(frequency.size, selected.size)
        else:
            power = np.empty((frequency.size, 0), dtype=np.float64)
        if depth_values.size == 0:
            raise ValueError("20 Hz AIS 船舶源产品没有有效声源深度。")
        source_depth = float(np.nanmedian(depth_values))

    return SpectralSourceSelection(
        product_path=path,
        time_index=0,
        time_utc=model_time_utc,
        receiver_lon=float(receiver_lon),
        receiver_lat=float(receiver_lat),
        radius_km=float(radius_km),
        source_depth_m=source_depth,
        longitude=longitude[selected],
        latitude=latitude[selected],
        distance_km=selected_distance,
        bearing_deg=_bearing_deg(longitude[selected], latitude[selected], receiver_lon, receiver_lat),
        frequency_hz=frequency,
        cell_source_power_linear=power,
        model="JOMOPANS_ECHO_AIS_20Hz",
    )


def load_current_sources(
    environment_product: Path,
    shipping_product: Path,
    requested_time: Any,
    receiver_lon: float,
    receiver_lat: float,
    radius_km: float,
    *,
    model_time_utc: str = MODEL_TIME_UTC,
    maximum_surface_cells: int = 441,
    maximum_shipping_cells: int = 5000,
) -> dict[str, SpectralSourceSelection]:
    model_time = validate_model_time(requested_time, model_time_utc)
    environment_path = Path(environment_product)
    shipping_path = Path(shipping_product)
    if not environment_path.is_file():
        raise FileNotFoundError(f"20 Hz 风雨源产品不存在：{environment_path}")
    if not shipping_path.is_file():
        raise FileNotFoundError(f"20 Hz AIS 船舶源产品不存在：{shipping_path}")
    wind, rain = _read_regular_components(
        environment_path,
        model_time,
        receiver_lon,
        receiver_lat,
        radius_km,
        maximum_surface_cells,
    )
    shipping = _read_shipping(
        shipping_path,
        model_time,
        receiver_lon,
        receiver_lat,
        radius_km,
        maximum_shipping_cells,
    )
    if not np.array_equal(wind.frequency_hz, shipping.frequency_hz):
        raise ValueError("20 Hz 风雨源产品与 AIS 船舶源产品的频率轴不一致。")
    expected = native_frequency_axis(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
    if not np.array_equal(wind.frequency_hz, expected):
        raise ValueError("源产品频率轴不是 20–1000 Hz、每 20 Hz 一个频点。")
    return {"wind": wind, "rain": rain, "shipping": shipping}


def source_power_at(selection: SpectralSourceSelection, frequency_hz: float) -> np.ndarray:
    hits = np.flatnonzero(np.isclose(selection.frequency_hz, float(frequency_hz), atol=1.0e-7))
    if hits.size != 1:
        raise ValueError(f"源产品中没有唯一的 {frequency_hz:g} Hz 频点。")
    return np.asarray(selection.cell_source_power_linear[int(hits[0])], dtype=np.float64)


def propagated_level_db(
    selection: SpectralSourceSelection,
    frequency_hz: float,
    tl_path: Path,
    *,
    source_depth_m: float,
) -> tuple[float, int]:
    if selection.source_count == 0:
        return -300.0, 0
    source_power = source_power_at(selection, frequency_hz)
    transmission_loss = sample_source_transmission_loss(
        selection, tl_path, source_depth_m=float(source_depth_m)
    )
    valid = np.isfinite(source_power) & (source_power > 0.0) & np.isfinite(transmission_loss)
    if not np.any(valid):
        return -300.0, 0
    receiver_power = float(
        np.sum(source_power[valid] * np.power(10.0, -transmission_loss[valid] / 10.0))
    )
    if not math.isfinite(receiver_power) or receiver_power <= 0.0:
        return -300.0, 0
    return 10.0 * math.log10(receiver_power), int(np.count_nonzero(valid))
