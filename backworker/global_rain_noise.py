"""构建可重复生成的全球 ERA5 雨噪声非均匀表面声源场。

输入的 ``tp`` 和 ``sf`` 字段是以米为单位的一小时累计水深。液态降雨率按
``max(tp - sf, 0) * 1000`` 计算，单位为 mm/h。输出止于声源建模；接收点
噪声仍需结合兼容的传播损失场，并按论文方法进行非相干功率叠加。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .global_wind_noise import (
    EPOCH_UNITS,
    PROJECT_ROOT,
    _axis,
    _canonical_axes,
    _find_variable,
    _json,
    _time_values,
    _variable,
    load_ocean_mask,
    nc_path,
    spherical_cell_area,
)
from .services.environment_noise import rain_surface_source_level_db


MODEL_VERSION = "global_era5_rain_surface_sources_v2"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "global_rain_noise.json"
SOURCE_NORMALIZATION = (
    "Published unit-area rain source level SLR=51+10log10(R_mm_h); "
    "cell power equals linear source density times spherical cell area"
)

LIMITATIONS = (
    "The published source-level relation is applied only over its 500-6400 Hz study band.",
    "Rain sources are represented at one-quarter acoustic wavelength below the sea surface.",
    "Liquid rain is approximated as max(total precipitation - snowfall water equivalent, 0).",
    "Sea-ice attenuation or shielding is not applied; output is potential open-water rain noise.",
    "Cells outside the configured ocean-mask latitude coverage are invalid, not zero-noise ocean.",
    "Receiver PSD requires a compatible propagation-loss calculation and linear-power summation.",
)


@dataclass(frozen=True)
class RainRecord:
    path: Path
    time_index: int
    epoch_seconds: int
    time_utc: str
    time_name: str
    latitude_name: str
    longitude_name: str
    total_precipitation_name: str
    snowfall_name: str
    latitude_order: np.ndarray
    longitude_order: np.ndarray


def _validate_precipitation_variable(
    dataset: Dataset,
    name: str,
    time_name: str,
    latitude_name: str,
    longitude_name: str,
) -> None:
    variable = dataset.variables[name]
    required = {
        dataset.variables[time_name].dimensions[0],
        dataset.variables[latitude_name].dimensions[0],
        dataset.variables[longitude_name].dimensions[0],
    }
    if variable.ndim != 3 or set(variable.dimensions) != required:
        raise ValueError(
            f"{name} 维度必须由时间、纬度、经度组成，实际为 {variable.dimensions}。"
        )
    units = str(getattr(variable, "units", "")).lower().replace(" ", "")
    if units not in {"m", "mofwaterequivalent", "metre", "meter"}:
        raise ValueError(
            f"{name} 单位必须是米水当量，实际为 {getattr(variable, 'units', '')!r}。"
        )
    step_type = str(getattr(variable, "GRIB_stepType", "")).strip().lower()
    if step_type and step_type != "accum":
        raise ValueError(f"{name} 必须是累计量，GRIB_stepType 实际为 {step_type!r}。")


def discover_rain_records(
    data_dir: Path,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, list[RainRecord]]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到 ERA5 降水文件：{data_dir / pattern}")

    reference_latitude: np.ndarray | None = None
    reference_longitude: np.ndarray | None = None
    records: list[RainRecord] = []
    for path in paths:
        with Dataset(nc_path(path), "r") as dataset:
            time_name = _find_variable(dataset, ("valid_time", "time"), "时间")
            latitude_name = _find_variable(dataset, ("latitude", "lat"), "纬度")
            longitude_name = _find_variable(dataset, ("longitude", "lon"), "经度")
            total_name = _find_variable(dataset, ("tp", "total_precipitation"), "总降水")
            snowfall_name = _find_variable(dataset, ("sf", "snowfall"), "降雪")
            latitude, longitude, latitude_order, longitude_order = _canonical_axes(
                _axis(dataset, latitude_name), _axis(dataset, longitude_name)
            )
            _validate_precipitation_variable(
                dataset, total_name, time_name, latitude_name, longitude_name
            )
            _validate_precipitation_variable(
                dataset, snowfall_name, time_name, latitude_name, longitude_name
            )
            seconds, labels = _time_values(dataset, time_name)
            if reference_latitude is None:
                reference_latitude, reference_longitude = latitude, longitude
            elif not (
                reference_latitude.shape == latitude.shape
                and reference_longitude is not None
                and reference_longitude.shape == longitude.shape
                and np.allclose(reference_latitude, latitude, rtol=0.0, atol=1.0e-10)
                and np.allclose(reference_longitude, longitude, rtol=0.0, atol=1.0e-10)
            ):
                raise ValueError(f"降水文件网格不一致：{path.name}")
            for index, (epoch, label) in enumerate(zip(seconds, labels)):
                records.append(
                    RainRecord(
                        path=path,
                        time_index=index,
                        epoch_seconds=int(epoch),
                        time_utc=label,
                        time_name=time_name,
                        latitude_name=latitude_name,
                        longitude_name=longitude_name,
                        total_precipitation_name=total_name,
                        snowfall_name=snowfall_name,
                        latitude_order=latitude_order,
                        longitude_order=longitude_order,
                    )
                )

    records.sort(key=lambda item: item.epoch_seconds)
    duplicates = [
        records[index].time_utc
        for index in range(1, len(records))
        if records[index - 1].epoch_seconds == records[index].epoch_seconds
    ]
    if duplicates:
        raise ValueError(f"ERA5 降水数据存在重复有效时次：{sorted(set(duplicates))}")
    assert reference_latitude is not None and reference_longitude is not None
    return reference_latitude, reference_longitude, records


def _read_field(dataset: Dataset, record: RainRecord, name: str) -> np.ndarray:
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


def read_precipitation(record: RainRecord) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path(record.path), "r") as dataset:
        total = _read_field(dataset, record, record.total_precipitation_name)
        snowfall = _read_field(dataset, record, record.snowfall_name)
    if not np.any(np.isfinite(total) & np.isfinite(snowfall)):
        raise ValueError(f"降水时次 {record.time_utc} 没有任何有效 tp/sf 数据。")
    return total, snowfall


def liquid_rain_rate_mm_h(
    total_precipitation_m: np.ndarray,
    snowfall_water_equivalent_m: np.ndarray,
    accumulation_hours: float,
) -> np.ndarray:
    if not math.isfinite(float(accumulation_hours)) or float(accumulation_hours) <= 0.0:
        raise ValueError("accumulation_hours 必须是正有限数。")
    total = np.asarray(total_precipitation_m, dtype=np.float64)
    snowfall = np.asarray(snowfall_water_equivalent_m, dtype=np.float64)
    return np.maximum(total - snowfall, 0.0) * (1000.0 / float(accumulation_hours))


def _fingerprint(config: dict[str, Any], records: list[RainRecord], mask_meta: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(mask_meta, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for record in records:
        stat = record.path.stat()
        digest.update(str(record.path.resolve()).encode("utf-8"))
        digest.update(
            f"{stat.st_size}:{stat.st_mtime_ns}:{record.time_index}:{record.epoch_seconds}".encode("ascii")
        )
    return digest.hexdigest()


def build_global_rain_sources(
    output_path: Path,
    config: dict[str, Any],
    data_dir: Path,
    requested_epoch_seconds: int | None = None,
) -> dict[str, Any]:
    pattern = str(config.get("rain_file_glob") or "ERA5-*_rain_sf.nc")
    latitude, longitude, records = discover_rain_records(data_dir, pattern)
    if requested_epoch_seconds is not None:
        wanted = int(requested_epoch_seconds)
        records = [record for record in records if record.epoch_seconds == wanted]
        if not records:
            requested_label = datetime.fromtimestamp(wanted, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            raise FileNotFoundError(
                f"本地 ERA5 降水场没有请求时次 {requested_label}，无法生成雨噪声源网格。"
            )
    frequencies = np.asarray(config.get("frequency_hz"), dtype=np.float64)
    if (
        frequencies.ndim != 1
        or frequencies.size < 2
        or frequencies.size > 1000
        or not np.all(np.isfinite(frequencies))
        or np.any(np.diff(frequencies) <= 0.0)
        or frequencies[0] < 500.0
        or frequencies[-1] > 6400.0
    ):
        raise ValueError("频率必须是 500–6400 Hz 内严格递增的 2–1000 个数值。")
    accumulation_hours = float(config.get("accumulation_hours", 1.0))
    if not math.isfinite(accumulation_hours) or accumulation_hours <= 0.0:
        raise ValueError("accumulation_hours 必须是正有限数。")
    minimum_rain_rate = float(config.get("minimum_rain_rate_mm_h", 0.1))
    if not math.isfinite(minimum_rain_rate) or minimum_rain_rate <= 0.0:
        raise ValueError("minimum_rain_rate_mm_h 必须是正有限数。")
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

    chunk_latitude = max(1, min(int(config.get("chunk_latitude_count", 48)), latitude.size))
    chunk_longitude = max(1, min(int(config.get("chunk_longitude_count", 360)), longitude.size))
    summaries: list[dict[str, Any]] = []
    try:
        with Dataset(nc_path(temporary), "w", format="NETCDF4") as dataset:
            dataset.Conventions = "CF-1.8"
            dataset.model_version = MODEL_VERSION
            dataset.model_status = "published_unit_area_surface_source"
            dataset.run_fingerprint = run_id
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            dataset.source_normalization = SOURCE_NORMALIZATION
            dataset.rain_rate_formula = "max(tp_m - sf_water_equivalent_m, 0) * 1000 / accumulation_hours"
            dataset.rain_source_level_formula = "SLR=51+10*log10(R_mm_h)"
            dataset.rain_source_model_reference = (
                "Liu Yunfeng et al. (2022), equations (5) and (8), citing Urick"
            )
            dataset.accumulation_hours = accumulation_hours
            dataset.minimum_rain_rate_mm_h = minimum_rain_rate
            dataset.source_depth_rule = "quarter_wavelength_c_over_4f"
            dataset.source_depth_reference_sound_speed_mps = 1500.0
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
            latitude_var = _variable(dataset, "latitude", "f8", ("latitude",), units="degrees_north")
            latitude_var.standard_name = "latitude"
            latitude_var[:] = latitude
            longitude_var = _variable(dataset, "longitude", "f8", ("longitude",), units="degrees_east")
            longitude_var.standard_name = "longitude"
            longitude_var[:] = longitude
            frequency_var = _variable(dataset, "frequency_hz", "f8", ("frequency",), units="Hz")
            frequency_var.standard_name = "sound_frequency"
            frequency_var[:] = frequencies
            area_var = _variable(
                dataset,
                "cell_area_m2",
                "f8",
                ("latitude", "longitude"),
                units="m2",
                chunksizes=(chunk_latitude, chunk_longitude),
            )
            area_var.long_name = "spherical grid-cell area"
            area_var[:, :] = area
            ocean_var = _variable(
                dataset,
                "ocean_mask",
                "i1",
                ("latitude", "longitude"),
                units="1",
                chunksizes=(chunk_latitude, chunk_longitude),
            )
            ocean_var.long_name = "ocean cell supported by configured static mask"
            ocean_var[:, :] = ocean.astype(np.int8)
            coverage_var = _variable(
                dataset,
                "ocean_mask_coverage",
                "i1",
                ("latitude", "longitude"),
                units="1",
                chunksizes=(chunk_latitude, chunk_longitude),
            )
            coverage_var.long_name = "target cell lies inside source mask latitude coverage"
            coverage_var[:, :] = mask_coverage.astype(np.int8)

            grid_chunks = (1, chunk_latitude, chunk_longitude)
            source_chunks = (1, 1, chunk_latitude, chunk_longitude)
            total_var = _variable(dataset, "total_precipitation_m", "f4", ("time", "latitude", "longitude"), units="m", fill_value=np.nan, chunksizes=grid_chunks)
            snowfall_var = _variable(dataset, "snowfall_water_equivalent_m", "f4", ("time", "latitude", "longitude"), units="m", fill_value=np.nan, chunksizes=grid_chunks)
            liquid_var = _variable(dataset, "liquid_precipitation_m", "f4", ("time", "latitude", "longitude"), units="m", fill_value=np.nan, chunksizes=grid_chunks)
            rain_rate_var = _variable(dataset, "rain_rate_mm_h", "f4", ("time", "latitude", "longitude"), units="mm h-1", fill_value=np.nan, chunksizes=grid_chunks)
            data_valid_var = _variable(dataset, "precipitation_data_valid_mask", "i1", ("time", "latitude", "longitude"), units="1", chunksizes=grid_chunks)
            source_valid_var = _variable(dataset, "source_valid_mask", "i1", ("time", "latitude", "longitude"), units="1", chunksizes=grid_chunks)
            source_valid_var.long_name = "ocean cells with rain rate at or above the configured source threshold"
            source_level_var = _variable(
                dataset,
                "rain_surface_source_level_db",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="dB re 1 uPa2 Hz-1 m-2 at 1 m",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            source_level_var.long_name = "published rain-noise source level per unit sea-surface area"
            density_var = _variable(
                dataset,
                "rain_source_density_linear",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="uPa2 Hz-1 at 1 m m-2",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            density_var.long_name = "rain source spectral density per unit sea-surface area"
            cell_var = _variable(
                dataset,
                "rain_cell_source_psd_linear",
                "f4",
                ("time", "frequency", "latitude", "longitude"),
                units="uPa2 Hz-1 at 1 m",
                fill_value=np.nan,
                chunksizes=source_chunks,
            )
            cell_var.long_name = "equivalent rain source spectrum integrated over each active ocean cell"

            for time_index, record in enumerate(records):
                print(f"Reading ERA5 precipitation {record.time_utc} from {record.path.name}", flush=True)
                total, snowfall = read_precipitation(record)
                finite = np.isfinite(total) & np.isfinite(snowfall)
                data_valid = ocean & finite
                liquid = np.maximum(total - snowfall, 0.0)
                rain_rate = liquid_rain_rate_mm_h(total, snowfall, accumulation_hours)
                active = data_valid & (rain_rate >= minimum_rain_rate)

                total_var[time_index, :, :] = np.where(data_valid, total, np.nan).astype(np.float32)
                snowfall_var[time_index, :, :] = np.where(data_valid, snowfall, np.nan).astype(np.float32)
                liquid_var[time_index, :, :] = np.where(data_valid, liquid, np.nan).astype(np.float32)
                rain_rate_var[time_index, :, :] = np.where(data_valid, rain_rate, np.nan).astype(np.float32)
                data_valid_var[time_index, :, :] = data_valid.astype(np.int8)
                source_valid_var[time_index, :, :] = active.astype(np.int8)

                for frequency_index, frequency in enumerate(frequencies):
                    source_level = np.full(rain_rate.shape, np.nan, dtype=np.float64)
                    source_level[active] = rain_surface_source_level_db(rain_rate[active])
                    density = np.full(rain_rate.shape, np.nan, dtype=np.float64)
                    density[data_valid] = 0.0
                    density[active] = np.power(10.0, source_level[active] / 10.0)
                    cell = density * area
                    source_level_var[time_index, frequency_index, :, :] = source_level.astype(np.float32)
                    density_var[time_index, frequency_index, :, :] = density.astype(np.float32)
                    cell_var[time_index, frequency_index, :, :] = cell.astype(np.float32)

                valid_values = rain_rate[data_valid]
                active_values = rain_rate[active]
                summaries.append(
                    {
                        "time_utc": record.time_utc,
                        "input_file": str(record.path.resolve()),
                        "input_time_index": record.time_index,
                        "valid_ocean_data_cells": int(np.count_nonzero(data_valid)),
                        "active_source_cells": int(np.count_nonzero(active)),
                        "rain_rate_mm_h": {
                            "min": float(np.min(valid_values)),
                            "max": float(np.max(valid_values)),
                            "mean": float(np.mean(valid_values)),
                        },
                        "active_rain_rate_mm_h": None
                        if active_values.size == 0
                        else {
                            "min": float(np.min(active_values)),
                            "max": float(np.max(active_values)),
                            "mean": float(np.mean(active_values)),
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
        "rain_file_glob": pattern,
        "time_count": len(records),
        "times_utc": [record.time_utc for record in records],
        "accumulation_hours": accumulation_hours,
        "minimum_rain_rate_mm_h": minimum_rain_rate,
        "frequency_count": int(frequencies.size),
        "frequency_hz": frequencies.tolist(),
        "grid_shape": [int(latitude.size), int(longitude.size)],
        "grid_cell_count": int(latitude.size * longitude.size),
        "ocean_mask_cells": int(np.count_nonzero(ocean)),
        "ocean_mask_coverage_cells": int(np.count_nonzero(mask_coverage)),
        "ocean_mask_area_m2": float(np.sum(area[ocean])),
        "ocean_mask": mask_meta,
        "source_normalization": SOURCE_NORMALIZATION,
        "source_depth_rule": "quarter_wavelength_c_over_4f",
        "source_depth_reference_sound_speed_mps": 1500.0,
        "snapshots": summaries,
        "limitations": list(LIMITATIONS),
        "receiver_psd_status": "not_computed_source_model_only",
    }


def run(config_path: Path, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"结果目录非空，请指定新目录以保留已有结果：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_global_rain_sources(output_dir / "global_rain_noise_sources.nc", config, data_dir)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = f"""# 全球 ERA5 雨噪声源建模

- 有效时次：{summary['time_count']}
- 时间：{', '.join(summary['times_utc'])}
- ERA5 网格：{summary['grid_shape'][0]} × {summary['grid_shape'][1]}
- 降水累计时长：{summary['accumulation_hours']} h
- 活动源雨强阈值：{summary['minimum_rain_rate_mm_h']} mm/h
- 频率数：{summary['frequency_count']}
- 海洋掩膜单元：{summary['ocean_mask_cells']}
- 接收点噪声：未计算（本产品仅到声源建模阶段）

## 输出

`global_rain_noise_sources.nc` 包含 tp、sf、液态降水量、小时雨强、海洋掩膜、
单元面积、活动源掩膜、论文单位面积源级、单位面积源谱和单元积分源谱。

## 限制

{chr(10).join('- ' + item for item in LIMITATIONS)}
"""
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("output_file", "time_count", "times_utc", "grid_shape", "ocean_mask_cells")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="自动扫描 ERA5 tp/sf 并生成全球雨噪声等效面源产品")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "HYdata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "jobs" / ("global_rain_noise_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    arguments = parser.parse_args()
    run(arguments.config, arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
