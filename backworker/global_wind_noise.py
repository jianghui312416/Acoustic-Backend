"""构建可重复生成的全球 ERA5 10 m 风噪声等效声源场。

该命令扫描所有匹配 ``wind_file_glob`` 的文件。时间从 NetCDF 坐标读取，
而不是根据文件名推断，因此添加新的 ERA5 文件后再次运行命令，即可生成
包含新时次的多时次产品。

本模块特意止于声源建模，不计算传播损失或接收点噪声，也不会隐式添加降雨、
船舶、海浪、生物或冰裂噪声分量。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
from netCDF4 import Dataset, date2num, num2date

EARTH_RADIUS_M = 6371008.8


def nc_path(path):
    """兼容 Windows NetCDF C 库不能直接打开中文绝对路径的情况。"""
    try:
        return os.path.relpath(Path(path).resolve())
    except ValueError:
        return str(path)


# netCDF4 1.7.x 与 NumPy 2.5 搭配使用时，内部会对 ndarray.shape 赋值。
# 该操作的结果仍然正确；仅屏蔽这一已知依赖项警告，避免影响正常模型运行日志的可读性。
warnings.filterwarnings(
    "ignore",
    message=r"Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)


MODEL_VERSION = "global_era5_wind_surface_sources_v1"
SOURCE_NORMALIZATION = "vertical_dipole_1m_equivalent_uncalibrated"
EPOCH_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "global_wind_noise.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIMITATIONS = (
    "The Wenz-type wind curve is an empirical reference PSD, not a measured source level.",
    "Equivalent source density uses an explicit reference transfer integral and remains uncalibrated.",
    "Sea-ice attenuation or shielding is not applied; output is potential open-water wind noise.",
    "Cells outside the configured ocean-mask latitude coverage are invalid, not zero-noise ocean.",
    "Receiver PSD requires a compatible propagation-loss calculation and linear-power summation.",
)


@dataclass(frozen=True)
class WindRecord:
    path: Path
    time_index: int
    epoch_seconds: int
    time_utc: str
    time_name: str
    latitude_name: str
    longitude_name: str
    u_name: str
    v_name: str
    latitude_order: np.ndarray
    longitude_order: np.ndarray


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"配置文件必须是 JSON 对象：{path}")
    return value


def _find_variable(dataset: Dataset, candidates: Iterable[str], label: str) -> str:
    by_lower = {str(name).lower(): str(name) for name in dataset.variables}
    for candidate in candidates:
        hit = by_lower.get(str(candidate).lower())
        if hit:
            return hit
    raise ValueError(f"NetCDF 缺少{label}变量，候选名：{list(candidates)}")


def _axis(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    if variable.ndim != 1:
        raise ValueError(f"坐标 {name} 必须是一维数组。")
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"坐标 {name} 存在缺测或点数不足。")
    return values


def _canonical_axes(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_order = np.argsort(latitude)
    lat = np.asarray(latitude[lat_order], dtype=np.float64)
    lon_normalized = np.mod(longitude, 360.0)
    lon_order = np.argsort(lon_normalized)
    lon = np.asarray(lon_normalized[lon_order], dtype=np.float64)
    if np.any(np.diff(lat) <= 0.0) or np.any(np.diff(lon) <= 0.0):
        raise ValueError("经纬度坐标必须互不重复并可严格排序。")
    if lat[0] < -90.0 or lat[-1] > 90.0 or lon[0] < 0.0 or lon[-1] >= 360.0:
        raise ValueError("经纬度坐标超出有效范围。")
    return lat, lon, lat_order, lon_order


def _time_values(dataset: Dataset, name: str) -> tuple[np.ndarray, list[str]]:
    variable = dataset.variables[name]
    if not getattr(variable, "units", ""):
        raise ValueError(f"时间变量 {name} 缺少 units。")
    raw = np.asarray(variable[:]).reshape(-1)
    calendar = getattr(variable, "calendar", "standard")
    dates = num2date(raw, variable.units, calendar=calendar)
    seconds = np.asarray(date2num(dates, EPOCH_UNITS, calendar="proleptic_gregorian"), dtype=np.float64)
    rounded = np.rint(seconds).astype(np.int64)
    if not np.allclose(seconds, rounded, rtol=0.0, atol=1.0e-6):
        raise ValueError("当前输出协议要求有效时间落在整秒。")
    labels = [datetime.fromtimestamp(int(value), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for value in rounded]
    return rounded, labels


def _validate_wind_variable(dataset: Dataset, name: str, time_name: str, lat_name: str, lon_name: str) -> None:
    variable = dataset.variables[name]
    required = {
        dataset.variables[time_name].dimensions[0],
        dataset.variables[lat_name].dimensions[0],
        dataset.variables[lon_name].dimensions[0],
    }
    if variable.ndim != 3 or set(variable.dimensions) != required:
        raise ValueError(
            f"{name} 维度必须由时间、纬度、经度组成，实际为 {variable.dimensions}。"
        )
    units = str(getattr(variable, "units", "")).lower().replace(" ", "")
    if units not in {"m/s", "ms-1", "ms**-1", "m.s-1"}:
        raise ValueError(f"{name} 单位必须是 m/s，实际为 {getattr(variable, 'units', '')!r}。")


def discover_wind_records(data_dir: Path, pattern: str) -> tuple[np.ndarray, np.ndarray, list[WindRecord]]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到 ERA5 风场文件：{data_dir / pattern}")

    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    records: list[WindRecord] = []
    for path in paths:
        with Dataset(nc_path(path), "r") as dataset:
            time_name = _find_variable(dataset, ("valid_time", "time"), "时间")
            lat_name = _find_variable(dataset, ("latitude", "lat"), "纬度")
            lon_name = _find_variable(dataset, ("longitude", "lon"), "经度")
            u_name = _find_variable(dataset, ("u10", "10u"), "10 米 U 风")
            v_name = _find_variable(dataset, ("v10", "10v"), "10 米 V 风")
            lat, lon, lat_order, lon_order = _canonical_axes(_axis(dataset, lat_name), _axis(dataset, lon_name))
            _validate_wind_variable(dataset, u_name, time_name, lat_name, lon_name)
            _validate_wind_variable(dataset, v_name, time_name, lat_name, lon_name)
            seconds, labels = _time_values(dataset, time_name)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not (
                reference_lat.shape == lat.shape
                and reference_lon is not None
                and reference_lon.shape == lon.shape
                and np.allclose(reference_lat, lat, rtol=0.0, atol=1.0e-10)
                and np.allclose(reference_lon, lon, rtol=0.0, atol=1.0e-10)
            ):
                raise ValueError(f"风场文件网格不一致：{path.name}")
            for index, (epoch, label) in enumerate(zip(seconds, labels)):
                records.append(
                    WindRecord(
                        path=path,
                        time_index=index,
                        epoch_seconds=int(epoch),
                        time_utc=label,
                        time_name=time_name,
                        latitude_name=lat_name,
                        longitude_name=lon_name,
                        u_name=u_name,
                        v_name=v_name,
                        latitude_order=lat_order,
                        longitude_order=lon_order,
                    )
                )

    records.sort(key=lambda item: item.epoch_seconds)
    duplicates = [
        records[index].time_utc
        for index in range(1, len(records))
        if records[index - 1].epoch_seconds == records[index].epoch_seconds
    ]
    if duplicates:
        raise ValueError(f"ERA5 风场存在重复有效时次：{sorted(set(duplicates))}")
    assert reference_lat is not None and reference_lon is not None
    return reference_lat, reference_lon, records


def _read_component(dataset: Dataset, record: WindRecord, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    time_dimension = dataset.variables[record.time_name].dimensions[0]
    latitude_dimension = dataset.variables[record.latitude_name].dimensions[0]
    longitude_dimension = dataset.variables[record.longitude_name].dimensions[0]
    selectors: list[Any] = []
    remaining: list[str] = []
    for dimension in variable.dimensions:
        if dimension == time_dimension:
            selectors.append(record.time_index)
        else:
            selectors.append(slice(None))
            remaining.append(dimension)
    values = np.asarray(np.ma.filled(variable[tuple(selectors)], np.nan), dtype=np.float64)
    if remaining == [latitude_dimension, longitude_dimension]:
        pass
    elif remaining == [longitude_dimension, latitude_dimension]:
        values = values.T
    else:
        raise ValueError(f"{name} 无法转换为 [latitude, longitude]：{remaining}")
    return values[np.ix_(record.latitude_order, record.longitude_order)]


def read_wind(record: WindRecord) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path(record.path), "r") as dataset:
        u10 = _read_component(dataset, record, record.u_name)
        v10 = _read_component(dataset, record, record.v_name)
    finite = np.isfinite(u10) & np.isfinite(v10)
    if not np.any(finite):
        raise ValueError(f"风场时次 {record.time_utc} 没有任何有效 u10/v10 数据。")
    return u10, v10


def _nearest_indices(axis: np.ndarray, query: np.ndarray) -> np.ndarray:
    order = np.argsort(axis)
    sorted_axis = np.asarray(axis[order], dtype=np.float64)
    position = np.clip(np.searchsorted(sorted_axis, query), 1, len(sorted_axis) - 1)
    left = position - 1
    choose_right = np.abs(query - sorted_axis[position]) < np.abs(query - sorted_axis[left])
    return order[np.where(choose_right, position, left)]


def _nearest_longitude_indices(axis: np.ndarray, query: np.ndarray) -> np.ndarray:
    normalized = np.mod(axis, 360.0)
    order = np.argsort(normalized)
    sorted_axis = normalized[order]
    extended = np.concatenate(([sorted_axis[-1] - 360.0], sorted_axis, [sorted_axis[0] + 360.0]))
    extended_order = np.concatenate(([order[-1]], order, [order[0]]))
    q = np.mod(query, 360.0)
    position = np.clip(np.searchsorted(extended, q), 1, len(extended) - 1)
    left = position - 1
    choose_right = np.abs(q - extended[position]) < np.abs(q - extended[left])
    return extended_order[np.where(choose_right, position, left)]


def load_ocean_mask(
    data_dir: Path,
    specification: dict[str, Any],
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    relative = str(specification.get("file") or "").strip()
    if not relative:
        raise ValueError("配置缺少 ocean_mask.file；不允许把 ERA5 陆地风场当成海面声源。")
    path = Path(relative)
    if not path.is_absolute():
        path = data_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"海洋掩膜文件不存在：{path}")

    with Dataset(nc_path(path), "r") as dataset:
        lat_name = str(specification.get("latitude_variable") or "").strip() or _find_variable(
            dataset, ("latitude", "lat"), "掩膜纬度"
        )
        lon_name = str(specification.get("longitude_variable") or "").strip() or _find_variable(
            dataset, ("longitude", "lon"), "掩膜经度"
        )
        variable_name = str(specification.get("variable") or "").strip()
        if variable_name not in dataset.variables:
            raise ValueError(f"海洋掩膜变量不存在：{variable_name!r}")
        source_lat = _axis(dataset, lat_name)
        source_lon = _axis(dataset, lon_name)
        lat_dimension = dataset.variables[lat_name].dimensions[0]
        lon_dimension = dataset.variables[lon_name].dimensions[0]
        variable = dataset.variables[variable_name]
        selectors: list[Any] = []
        remaining: list[str] = []
        for dimension in variable.dimensions:
            if dimension == lat_dimension or dimension == lon_dimension:
                selectors.append(slice(None))
                remaining.append(dimension)
            elif "time" in dimension.lower():
                selectors.append(int(specification.get("time_index", 0)))
            elif "depth" in dimension.lower() or dimension.lower() in {"lev", "level"}:
                selectors.append(int(specification.get("depth_index", 0)))
            elif len(dataset.dimensions[dimension]) == 1:
                selectors.append(0)
            else:
                raise ValueError(f"掩膜变量的维度 {dimension} 无法自动选层。")
        raw = variable[tuple(selectors)]
        values = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
        if remaining == [lat_dimension, lon_dimension]:
            pass
        elif remaining == [lon_dimension, lat_dimension]:
            values = values.T
        else:
            raise ValueError(f"掩膜变量无法转换为 [latitude, longitude]：{remaining}")

    rule = str(specification.get("ocean_rule") or "finite").strip().lower()
    if rule == "finite":
        source_ocean = np.isfinite(values)
    elif rule == "less_than":
        threshold = float(specification.get("threshold", 0.5))
        source_ocean = np.isfinite(values) & (values < threshold)
    elif rule == "greater_than":
        threshold = float(specification.get("threshold", 0.5))
        source_ocean = np.isfinite(values) & (values > threshold)
    else:
        raise ValueError("ocean_mask.ocean_rule 仅支持 finite、less_than 或 greater_than。")

    lat_index = _nearest_indices(source_lat, target_latitude)
    lon_index = _nearest_longitude_indices(source_lon, target_longitude)
    coverage_lat = (target_latitude >= float(np.min(source_lat))) & (target_latitude <= float(np.max(source_lat)))
    coverage = np.broadcast_to(coverage_lat[:, None], (target_latitude.size, target_longitude.size)).copy()
    ocean = source_ocean[np.ix_(lat_index, lon_index)] & coverage
    provenance = {
        "path": str(path.resolve()),
        "variable": variable_name,
        "ocean_rule": rule,
        "source_latitude_range": [float(np.min(source_lat)), float(np.max(source_lat))],
        "outside_coverage_policy": "invalid_source_cell",
    }
    return ocean, coverage, provenance


def spherical_cell_area(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if np.any(np.diff(lat) <= 0.0) or np.any(np.diff(lon) <= 0.0):
        raise ValueError("计算面积时经纬度必须严格递增。")
    lat_edges = np.empty(lat.size + 1, dtype=np.float64)
    lat_edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    lat_edges[0] = -90.0
    lat_edges[-1] = 90.0
    if np.any(np.diff(lat_edges) <= 0.0):
        raise ValueError("纬度网格无法构建有效单元边界。")

    previous = np.mod(lon - np.roll(lon, 1), 360.0)
    following = np.mod(np.roll(lon, -1) - lon, 360.0)
    lon_width = 0.5 * (previous + following)
    if not math.isclose(float(np.sum(lon_width)), 360.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("经度网格未完整覆盖 360°，不能生成全球单元面积。")
    lat_factor = np.diff(np.sin(np.deg2rad(lat_edges)))
    return EARTH_RADIUS_M**2 * lat_factor[:, None] * np.deg2rad(lon_width)[None, :]


def wind_reference_psd_db(frequency_hz: float, wind_speed_mps: np.ndarray) -> np.ndarray:
    frequency_khz = float(frequency_hz) / 1000.0
    return (
        50.0
        + 7.5 * np.sqrt(wind_speed_mps)
        + 20.0 * np.log10(frequency_khz)
        - 40.0 * np.log10(frequency_khz + 0.4)
    )


def _variable(
    dataset: Dataset,
    name: str,
    dtype: str,
    dimensions: tuple[str, ...],
    *,
    units: str,
    fill_value: float | None = None,
    chunksizes: tuple[int, ...] | None = None,
):
    options: dict[str, Any] = {"zlib": True, "complevel": 4, "shuffle": True}
    if fill_value is not None:
        options["fill_value"] = fill_value
    if chunksizes is not None:
        options["chunksizes"] = chunksizes
    result = dataset.createVariable(name, dtype, dimensions, **options)
    result.units = units
    return result


def _fingerprint(config: dict[str, Any], records: list[WindRecord], mask_meta: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(mask_meta, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for record in records:
        stat = record.path.stat()
        digest.update(str(record.path.resolve()).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{record.time_index}:{record.epoch_seconds}".encode("ascii"))
    return digest.hexdigest()


def build_global_wind_sources(
    output_path: Path,
    config: dict[str, Any],
    data_dir: Path,
    requested_epoch_seconds: int | None = None,
) -> dict[str, Any]:
    pattern = str(config.get("wind_file_glob") or "ERA5-10m-*_ws.nc")
    latitude, longitude, records = discover_wind_records(data_dir, pattern)
    if requested_epoch_seconds is not None:
        wanted = int(requested_epoch_seconds)
        records = [record for record in records if record.epoch_seconds == wanted]
        if not records:
            requested_label = datetime.fromtimestamp(wanted, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            raise FileNotFoundError(
                f"本地 ERA5 风场没有请求时次 {requested_label}，无法生成风噪声源网格。"
            )
    frequencies = np.asarray(config.get("frequency_hz"), dtype=np.float64)
    if (
        frequencies.ndim != 1
        or frequencies.size < 2
        or frequencies.size > 1000
        or not np.all(np.isfinite(frequencies))
        or np.any(np.diff(frequencies) <= 0.0)
        or frequencies[0] < 500.0
        or frequencies[-1] > 50000.0
    ):
        raise ValueError("频率必须是 500–50000 Hz 内严格递增的 2–1000 个数值。")
    reference_integral = float(config.get("reference_transfer_integral_m2", math.pi))
    if not math.isfinite(reference_integral) or reference_integral <= 0.0:
        raise ValueError("reference_transfer_integral_m2 必须是正有限数。")
    source_depth = float(config.get("source_depth_m", 1.0))
    if not math.isfinite(source_depth) or source_depth <= 0.0:
        raise ValueError("source_depth_m 必须是正有限数。")

    ocean, mask_coverage, mask_meta = load_ocean_mask(
        data_dir,
        dict(config.get("ocean_mask") or {}),
        latitude,
        longitude,
    )
    area = spherical_cell_area(latitude, longitude)
    run_id = _fingerprint(config, records, mask_meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    chunk_lat = max(1, min(int(config.get("chunk_latitude_count", 48)), latitude.size))
    chunk_lon = max(1, min(int(config.get("chunk_longitude_count", 360)), longitude.size))
    summaries: list[dict[str, Any]] = []
    try:
        with Dataset(nc_path(temporary), "w", format="NETCDF4") as dataset:
            dataset.Conventions = "CF-1.8"
            dataset.model_version = MODEL_VERSION
            dataset.model_status = "uncalibrated_engineering_equivalent_source"
            dataset.run_fingerprint = run_id
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            dataset.source_normalization = SOURCE_NORMALIZATION
            dataset.wind_reference_formula = (
                "50 + 7.5*sqrt(U10_m_s) + 20*log10(f_kHz) - 40*log10(f_kHz+0.4)"
            )
            dataset.reference_transfer_integral_m2 = reference_integral
            dataset.source_depth_m = source_depth
            dataset.ocean_mask_provenance_json = json.dumps(mask_meta, ensure_ascii=False)
            dataset.limitations_json = json.dumps(LIMITATIONS, ensure_ascii=False)
            dataset.createDimension("time", len(records))
            dataset.createDimension("frequency", frequencies.size)
            dataset.createDimension("latitude", latitude.size)
            dataset.createDimension("longitude", longitude.size)

            time_var = _variable(dataset, "time", "i8", ("time",), units=EPOCH_UNITS)
            time_var.calendar = "proleptic_gregorian"
            time_var.standard_name = "time"
            time_var[:] = [record.epoch_seconds for record in records]
            lat_var = _variable(dataset, "latitude", "f8", ("latitude",), units="degrees_north")
            lat_var.standard_name = "latitude"
            lat_var[:] = latitude
            lon_var = _variable(dataset, "longitude", "f8", ("longitude",), units="degrees_east")
            lon_var.standard_name = "longitude"
            lon_var[:] = longitude
            freq_var = _variable(dataset, "frequency_hz", "f8", ("frequency",), units="Hz")
            freq_var.standard_name = "sound_frequency"
            freq_var[:] = frequencies
            area_var = _variable(
                dataset,
                "cell_area_m2",
                "f8",
                ("latitude", "longitude"),
                units="m2",
                chunksizes=(chunk_lat, chunk_lon),
            )
            area_var.long_name = "spherical grid-cell area"
            area_var[:, :] = area
            ocean_var = _variable(
                dataset,
                "ocean_mask",
                "i1",
                ("latitude", "longitude"),
                units="1",
                chunksizes=(chunk_lat, chunk_lon),
            )
            ocean_var.long_name = "ocean cell supported by configured static mask"
            ocean_var[:, :] = ocean.astype(np.int8)
            coverage_var = _variable(
                dataset,
                "ocean_mask_coverage",
                "i1",
                ("latitude", "longitude"),
                units="1",
                chunksizes=(chunk_lat, chunk_lon),
            )
            coverage_var.long_name = "target cell lies inside source mask latitude coverage"
            coverage_var[:, :] = mask_coverage.astype(np.int8)

            grid_chunks = (1, chunk_lat, chunk_lon)
            source_chunks = (1, 1, chunk_lat, chunk_lon)
            u_var = _variable(dataset, "u10", "f4", ("time", "latitude", "longitude"), units="m s-1", fill_value=np.nan, chunksizes=grid_chunks)
            v_var = _variable(dataset, "v10", "f4", ("time", "latitude", "longitude"), units="m s-1", fill_value=np.nan, chunksizes=grid_chunks)
            speed_var = _variable(dataset, "wind_speed_mps", "f4", ("time", "latitude", "longitude"), units="m s-1", fill_value=np.nan, chunksizes=grid_chunks)
            valid_var = _variable(dataset, "source_valid_mask", "i1", ("time", "latitude", "longitude"), units="1", chunksizes=grid_chunks)
            reference_var = _variable(
                dataset,
                "wind_reference_psd_db",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="dB re 1 uPa2 Hz-1",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            reference_var.long_name = "empirical Wenz-type wind-noise reference power spectral density"
            density_var = _variable(
                dataset,
                "wind_source_density_linear",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="uPa2 Hz-1 at 1 m m-2",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            density_var.long_name = "uncalibrated equivalent wind source spectral density"
            cell_var = _variable(
                dataset,
                "wind_cell_source_psd_linear",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="uPa2 Hz-1 at 1 m",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            cell_var.long_name = "equivalent wind source spectrum integrated over each ocean cell"

            for time_index, record in enumerate(records):
                print(f"Reading ERA5 wind {record.time_utc} from {record.path.name}", flush=True)
                u10, v10 = read_wind(record)
                wind = np.hypot(u10, v10)
                valid = ocean & np.isfinite(wind)
                u_var[time_index, :, :] = np.where(valid, u10, np.nan).astype(np.float32)
                v_var[time_index, :, :] = np.where(valid, v10, np.nan).astype(np.float32)
                speed_var[time_index, :, :] = np.where(valid, wind, np.nan).astype(np.float32)
                valid_var[time_index, :, :] = valid.astype(np.int8)
                for frequency_index, frequency in enumerate(frequencies):
                    reference = wind_reference_psd_db(float(frequency), wind)
                    reference = np.where(valid, reference, np.nan)
                    density = np.power(10.0, reference / 10.0) / reference_integral
                    cell = density * area
                    reference_var[time_index, frequency_index, :, :] = reference.astype(np.float32)
                    density_var[time_index, frequency_index, :, :] = density.astype(np.float32)
                    cell_var[time_index, frequency_index, :, :] = cell.astype(np.float32)
                values = wind[valid]
                summaries.append(
                    {
                        "time_utc": record.time_utc,
                        "input_file": str(record.path.resolve()),
                        "input_time_index": record.time_index,
                        "valid_source_cells": int(np.count_nonzero(valid)),
                        "wind_speed_mps": {
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "mean": float(np.mean(values)),
                        },
                    }
                )
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "model_version": MODEL_VERSION,
        "run_fingerprint": run_id,
        "output_file": str(output_path.resolve()),
        "wind_file_glob": pattern,
        "time_count": len(records),
        "times_utc": [record.time_utc for record in records],
        "frequency_count": int(frequencies.size),
        "frequency_hz": frequencies.tolist(),
        "grid_shape": [int(latitude.size), int(longitude.size)],
        "grid_cell_count": int(latitude.size * longitude.size),
        "ocean_mask_cells": int(np.count_nonzero(ocean)),
        "ocean_mask_coverage_cells": int(np.count_nonzero(mask_coverage)),
        "ocean_mask_area_m2": float(np.sum(area[ocean])),
        "ocean_mask": mask_meta,
        "source_normalization": SOURCE_NORMALIZATION,
        "reference_transfer_integral_m2": reference_integral,
        "source_depth_m": source_depth,
        "snapshots": summaries,
        "limitations": list(LIMITATIONS),
        "receiver_psd_status": "not_computed_source_model_only",
    }


def run(config_path: Path, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"结果目录非空，请指定新目录以保留已有结果：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_global_wind_sources(output_dir / "global_wind_noise_sources.nc", config, data_dir)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = f"""# 全球 ERA5 风噪声源建模

- 有效时次：{summary['time_count']}
- 时间：{', '.join(summary['times_utc'])}
- ERA5 网格：{summary['grid_shape'][0]} × {summary['grid_shape'][1]}
- 频率数：{summary['frequency_count']}
- 海洋源单元：{summary['ocean_mask_cells']}
- 接收点噪声：未计算（本产品仅到声源建模阶段）

## 输出

`global_wind_noise_sources.nc` 包含 u10/v10、风速、海洋掩膜、单元面积、Wenz 参考 PSD、
等效单位面积源谱和单元积分源谱。新增 ERA5 时次文件后，重新运行命令即会自动扫描并按有效时间合并。

## 限制

{chr(10).join('- ' + item for item in LIMITATIONS)}
"""
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("output_file", "time_count", "times_utc", "grid_shape", "ocean_mask_cells")}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="自动扫描 ERA5 u10/v10 并生成全球风噪声等效面源产品")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "HYdata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "jobs" / ("global_wind_noise_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    arguments = parser.parse_args()
    run(arguments.config, arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
