"""分布式噪声源（风 / 降雨 / AIS 航运）网格建模的接口层共用编排。

把 ``AnalyseEnvironmentNoise`` 中数据驱动模式的源产品解析、局部源选取、
声传播作业编排抽取出来，供普通环境噪声接口与冰区环境噪声接口复用。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from ..global_rain_noise import build_global_rain_sources
from ..global_shipping_noise_ais import build_shipping_source_product
from ..global_wind_noise import build_global_wind_sources
from ..services.environment_noise_20hz import (
    MODEL_ID as CURRENT_20HZ_MODEL_ID,
    RAIN_MAX_FREQUENCY_HZ as CURRENT_RAIN_MAX_FREQUENCY_HZ,
    RAIN_MIN_FREQUENCY_HZ as CURRENT_RAIN_MIN_FREQUENCY_HZ,
    load_current_sources,
    native_frequency_axis,
    propagated_level_db as propagated_current_level_db,
    shipping_source_depth_m as current_shipping_source_depth_m,
    surface_source_depth_m as current_surface_source_depth_m,
)
from ..services.wind_noise_receiver import (
    RAIN_MODEL_MAX_FREQUENCY_HZ,
    RAIN_MODEL_MIN_FREQUENCY_HZ,
    load_local_rain_sources,
    load_local_shipping_sources,
    load_local_sources,
    propagated_liu2022_wind_level_db,
    propagated_rain_level_db,
    propagated_shipping_level_db,
    propagation_frequencies,
    resolve_source_product,
    rain_surface_source_depth_m,
    sample_wind_at_receiver,
    uniform_polar_wind_selection,
)
from ..services.hildebrand_wind_noise import hildebrand2021_psd_db, model_metadata
from ..services.vertical_array_directivity import source_grid_vertical_directivity
from .common import apply_fixed_acoustic_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_BUILD_LOCKS: dict[int, threading.Lock] = {}
_SOURCE_BUILD_LOCKS_GUARD = threading.Lock()

_PRODUCT_SPECS = {
    "wind": (
        "WIND_SOURCE_PRODUCT_GLOB",
        "jobs/global_wind_noise_*/global_wind_noise_sources*.nc",
        "风噪声",
    ),
    "rain": (
        "RAIN_SOURCE_PRODUCT_GLOB",
        "jobs/global_rain_noise_*/global_rain_noise_sources*.nc",
        "降雨噪声",
    ),
    "shipping": (
        "SHIPPING_SOURCE_PRODUCT_GLOB",
        "jobs/shipping_sources_ais_*_final/shipping_sources*.nc",
        "AIS 航运噪声",
    ),
}


def data_config(core):
    config = core.CONFIG.get("ENVIRONMENT_NOISE_DATA") or {}
    if not isinstance(config, dict):
        raise ValueError("data_config.json 中 ENVIRONMENT_NOISE_DATA 必须是对象。")
    return config


def product_paths(config, key, default_pattern):
    pattern = str(config.get(key) or default_pattern)
    return sorted(PROJECT_ROOT.glob(pattern))


def _project_path(value: Any, default: str) -> Path:
    path = Path(str(value or default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _requested_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("自动生成噪声源网格时必须传入 time。")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"无法解析请求时间 {value!r}，应使用 yyyy-MM-dd HH:mm:ss。") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _source_build_lock(epoch_seconds: int) -> threading.Lock:
    with _SOURCE_BUILD_LOCKS_GUARD:
        return _SOURCE_BUILD_LOCKS.setdefault(int(epoch_seconds), threading.Lock())


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}配置文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}配置文件必须是 JSON 对象：{path}")
    return value


def _resolve_configured_products(config, requested_time):
    resolved = {}
    errors = {}
    for name, (key, pattern, label) in _PRODUCT_SPECS.items():
        try:
            resolved[name] = resolve_source_product(
                product_paths(config, key, pattern), requested_time, label
            )
        except (FileNotFoundError, ValueError, IndexError) as exc:
            errors[name] = exc
    return resolved, errors


def _source_output_paths(config, requested: datetime) -> dict[str, Path]:
    tag = requested.strftime("%Y%m%dT%H%M%SZ")
    root = _project_path(config.get("SOURCE_CACHE_ROOT"), "jobs")
    return {
        "wind": root / f"global_wind_noise_{tag}" / f"global_wind_noise_sources_{tag}.nc",
        "rain": root / f"global_rain_noise_{tag}" / f"global_rain_noise_sources_{tag}.nc",
        "shipping": (
            root
            / f"shipping_sources_ais_{tag}_final"
            / f"shipping_sources_{tag}.nc"
        ),
    }


def _build_missing_source_products(
    config: Mapping[str, Any],
    requested: datetime,
    missing: set[str],
    *,
    wind_input_path: Path | None = None,
) -> None:
    epoch_seconds = int(requested.timestamp())
    outputs = _source_output_paths(config, requested)
    raw_data_dir = _project_path(config.get("SOURCE_RAW_DATA_DIR"), "HYdata")

    if "wind" in missing:
        wind_config = _read_json_object(
            _project_path(config.get("WIND_SOURCE_BUILD_CONFIG"), "config/global_wind_noise.json"),
            "风噪声源",
        )
        wind_data_dir = raw_data_dir
        if wind_input_path is not None:
            selected_wind = Path(wind_input_path).resolve()
            if not selected_wind.is_file():
                raise FileNotFoundError(f"已解析的 ERA5 风场文件不存在：{selected_wind}")
            wind_config["wind_file_glob"] = selected_wind.name
            wind_data_dir = selected_wind.parent
        build_global_wind_sources(
            outputs["wind"], wind_config, wind_data_dir,
            requested_epoch_seconds=epoch_seconds,
        )

    if "rain" in missing:
        rain_config = _read_json_object(
            _project_path(config.get("RAIN_SOURCE_BUILD_CONFIG"), "config/global_rain_noise.json"),
            "雨噪声源",
        )
        build_global_rain_sources(
            outputs["rain"], rain_config, raw_data_dir,
            requested_epoch_seconds=epoch_seconds,
        )

    if "shipping" in missing:
        shipping_config = _read_json_object(
            _project_path(
                config.get("SHIPPING_SOURCE_BUILD_CONFIG"),
                "config/global_shipping_noise_ais.json",
            ),
            "船舶噪声源",
        )
        shipping_config["time_utc"] = requested.strftime("%Y-%m-%dT%H:%M:%SZ")
        input_template = str(
            config.get("SHIPPING_RAW_INPUT_TEMPLATE")
            or (
                "HYdata/ais_{ais_tag}.jsonl/ais_{ais_tag}.jsonl"
            )
        )
        gfw_tag = requested.strftime("%Y%m%d_%HUTC")
        ais_tag = requested.strftime("%Y%m%d_%H")
        shipping_input = _project_path(
            input_template.format(
                gfw_tag=gfw_tag,
                ais_tag=ais_tag,
                time_tag=requested.strftime("%Y%m%dT%H%M%SZ"),
            ),
            "",
        )
        if not shipping_input.is_file():
            raise FileNotFoundError(
                f"本地没有请求时次 {shipping_config['time_utc']} 的 AIS JSONL 数据："
                f"{shipping_input}"
            )
        build_shipping_source_product(
            outputs["shipping"], shipping_config, shipping_input
        )


def ensure_source_products(config, requested_time, *, wind_input_path: Path | None = None):
    """命中指定时次的源网格缓存；缺失时由本地原始数据生成后再次解析。"""
    resolved, errors = _resolve_configured_products(config, requested_time)
    if not errors:
        return resolved
    if requested_time in (None, ""):
        raise next(iter(errors.values()))
    if not bool(config.get("ON_DEMAND_SOURCE_BUILD_ENABLED", True)):
        raise next(iter(errors.values()))

    requested = _requested_datetime(requested_time)
    epoch_seconds = int(requested.timestamp())
    with _source_build_lock(epoch_seconds):
        # 前一个并发请求可能已经完成了同一时次的缓存生成。
        resolved, errors = _resolve_configured_products(config, requested_time)
        if errors:
            _build_missing_source_products(
                config,
                requested,
                set(errors),
                wind_input_path=wind_input_path,
            )
        resolved, errors = _resolve_configured_products(config, requested_time)
        if errors:
            raise next(iter(errors.values()))
        return resolved


def propagation_request(normalised, selection, frequency_hz, config):
    return {
        "coordX": normalised["receiverLon"],
        "coordY": normalised["receiverLat"],
        # 声传播互易：模型中心声源代表真实接收点。
        "srcDepth": normalised["receiverDepthM"],
        "range": normalised["radiusKm"] / 1.852,
        "time": selection.time_utc.replace("T", " ").replace("Z", ""),
        "timeRes": 0,
        "rtime": selection.time_utc.replace("T", " ").replace("Z", ""),
        "rtimeRes": 0,
        "minFreq": max(1, int(round(float(frequency_hz)))),
        "maxFreq": max(1, int(round(float(frequency_hz)))),
        "row": int(config.get("TL_DEPTH_COUNT", 41)),
        "col": int(config.get("TL_RANGE_COUNT", 121)),
        "sector": int(config.get("SECTOR_COUNT", 8)),
    }


def build_current_20hz_levels(normalised, core):
    """使用本次 20 Hz 源产品并在每个原生频点运行经典 Bellhop。"""
    config = data_config(core)
    solver_setting = str(core.CONFIG.get("BELLHOP_CXX", "../bellhop.exe"))
    solver_name = Path(solver_setting).name.lower()
    if any(marker in solver_name for marker in ("bellhopcxx", "bellhopcuda", "bhc_")):
        raise core.PortalRequestError(
            "当前 20 Hz 环境噪声模型只允许经典 bellhop.exe，不允许 BellhopCXX/CUDA。"
        )
    environment_product = _project_path(
        config.get("CURRENT_20HZ_ENVIRONMENT_PRODUCT"),
        "output/environment_sources_20250102_12utc_20hz_v1/"
        "global_environment_source_spectrum_20hz.nc",
    )
    shipping_product = _project_path(
        config.get("CURRENT_20HZ_SHIPPING_PRODUCT"),
        "output/shipping_sources_ais_20261015_03_jomopans_20hz_v1/"
        "shipping_source_spectrum_20hz.nc",
    )
    try:
        selections = load_current_sources(
            environment_product,
            shipping_product,
            normalised.get("requestedTime"),
            normalised["receiverLon"],
            normalised["receiverLat"],
            normalised["radiusKm"],
            model_time_utc=str(
                config.get("CURRENT_20HZ_MODEL_TIME_UTC", "2025-01-02T12:00:00Z")
            ),
            maximum_surface_cells=int(config.get("MAX_WIND_SOURCE_CELLS", 441)),
            maximum_shipping_cells=int(config.get("MAX_SHIPPING_SOURCE_CELLS", 5000)),
        )
        frequencies = native_frequency_axis(
            normalised["minFreq"], normalised["maxFreq"]
        )
    except (OSError, ValueError, IndexError) as exc:
        raise core.PortalRequestError(str(exc)) from exc

    levels = {name: {} for name in selections}
    contributing_counts = {name: {} for name in selections}
    try:
        bottom_depths = np.asarray(
            core._env_bottom_depths(
                np.asarray([normalised["receiverLon"]], dtype=np.float64),
                np.asarray([normalised["receiverLat"]], dtype=np.float64),
                "EnvReanalysis",
                normalised.get("requestedTime"),
            ),
            dtype=np.float64,
        ).ravel()
        valid_bottom = bottom_depths[np.isfinite(bottom_depths) & (bottom_depths > 0.0)]
        if valid_bottom.size == 0:
            raise ValueError("接收点位于陆地或缺少有效水深。")
        water_depth_m = float(valid_bottom[0])
        if float(normalised["receiverDepthM"]) >= water_depth_m:
            raise ValueError(
                f"工作深度必须小于当前位置海深 {water_depth_m:.3f} 米。"
            )
    except (OSError, ValueError, IndexError) as exc:
        raise core.PortalRequestError(str(exc)) from exc
    def run_frequency(frequency_hz):
        frequency = float(frequency_hz)
        surface_depth = current_surface_source_depth_m(frequency, water_depth_m)
        shipping_depth = current_shipping_source_depth_m(
            selections["shipping"].source_depth_m, water_depth_m
        )
        request = propagation_request(
            normalised, selections["wind"], frequency, config
        )
        params, _, _, _ = core._portal_3d_params(request)
        # 用户指定本模型只使用经典 Bellhop；bellhop_config.json 当前实际
        # 指向 bellhop.exe，runner 会按经典 Fortran 参数形式调用它。
        apply_fixed_acoustic_model(params, "bellhop")
        depth_count = int(config.get("TL_DEPTH_COUNT", 41))
        range_count = int(config.get("CURRENT_20HZ_TL_RANGE_COUNT", 301))
        params.update({
            "freq_hz": frequency,
            "source_depth_m": float(normalised["receiverDepthM"]),
            "receiver_depth_min_m": min(surface_depth, shipping_depth),
            "receiver_depth_max_m": max(
                surface_depth + 1.0,
                shipping_depth + 1.0,
                float(normalised["receiverDepthM"]),
            ),
            "receiver_depth_auto_max": False,
            "receiver_depth_auto_scope": "area",
            "horizontal_tl_depth_m": min(surface_depth, shipping_depth),
            "tl_n_rd": depth_count,
            "tl_n_range": range_count,
            "n_rd": depth_count,
            "n_range": range_count,
            "wind_noise_reciprocity": True,
            "actual_receiver": {
                "longitude": normalised["receiverLon"],
                "latitude": normalised["receiverLat"],
                "depth_m": normalised["receiverDepthM"],
            },
        })
        job_id, job_dir, _ = core._run_portal_job(params)
        tl_path = Path(job_dir) / "tl3d_depth_theta_range.nc"
        component_results = {
            "wind": propagated_current_level_db(
                selections["wind"],
                frequency,
                tl_path,
                source_depth_m=surface_depth,
            ),
            "shipping": propagated_current_level_db(
                selections["shipping"],
                frequency,
                tl_path,
                source_depth_m=shipping_depth,
            ),
        }
        if CURRENT_RAIN_MIN_FREQUENCY_HZ <= frequency <= CURRENT_RAIN_MAX_FREQUENCY_HZ:
            component_results["rain"] = propagated_current_level_db(
                selections["rain"],
                frequency,
                tl_path,
                source_depth_m=surface_depth,
            )
        return {
            "frequency": frequency,
            "jobId": job_id,
            "components": component_results,
        }

    frequency_workers = max(
        1,
        min(
            int(config.get("CURRENT_20HZ_FREQUENCY_WORKERS", 3)),
            int(frequencies.size),
        ),
    )
    if frequency_workers == 1:
        frequency_results = [run_frequency(value) for value in frequencies]
    else:
        frequency_results = []
        with ThreadPoolExecutor(
            max_workers=frequency_workers,
            thread_name_prefix="environment-noise-20hz",
        ) as pool:
            futures = {pool.submit(run_frequency, value): float(value) for value in frequencies}
            try:
                for future in as_completed(futures):
                    frequency_results.append(future.result())
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    frequency_results.sort(key=lambda item: item["frequency"])
    job_ids = []
    propagation_jobs = []
    for result in frequency_results:
        frequency = float(result["frequency"])
        job_id = str(result["jobId"])
        job_ids.append(job_id)
        propagation_jobs.append({
            "jobId": job_id,
            "frequencyHz": round(frequency, 6),
            "model": "bellhop",
            "solver": "classic_fortran_bellhop",
        })
        for name, (level_db, contributing) in result["components"].items():
            levels[name][frequency] = level_db
            contributing_counts[name][f"{frequency:g}"] = contributing

    normalised["_current20HzModel"] = {
        "id": CURRENT_20HZ_MODEL_ID,
        "nativeFrequencyHz": [float(value) for value in frequencies],
        "frequencyStepHz": 20.0,
        "modelTimeUtc": selections["wind"].time_utc,
        "shippingTimeAssumption": (
            "AIS 2026-10-15 03 UTC spatial distribution is treated as "
            "2025-01-02 12 UTC"
        ),
        "sourceProducts": {
            "windRain": str(environment_product),
            "shipping": str(shipping_product),
        },
    }
    normalised["_current20HzSourceModels"] = {
        "wind": {
            "model": "Ainslie_2010_areic_point_dipole",
            "frequencyHz": [20.0, 1000.0],
            "frequencyStepHz": 20.0,
            "calibration": "published_areic_source_spectrum",
        },
        "rain": {
            "model": "Urick_amplitude_Ma_Nystuen_inspired_shape",
            "frequencyHz": [500.0, 1000.0],
            "frequencyStepHz": 20.0,
            "calibration": "engineering_hybrid",
        },
        "shipping": {
            "model": "JOMOPANS_ECHO_AIS_20Hz",
            "frequencyHz": [20.0, 1000.0],
            "frequencyStepHz": 20.0,
            "calibration": "semi_empirical_uncalibrated_for_this_AIS_sample",
        },
    }
    normalised["_hybridPropagation"] = {
        "model": "bellhop_all_native_20hz_frequencies",
        "selectionMode": "bellhop",
        "lowFrequencyModel": "bellhop",
        "highFrequencyModel": "bellhop",
        "frequencyInterpolation": "none_on_native_20hz_grid",
        "solver": "classic_fortran_bellhop",
        "frequencyWorkers": frequency_workers,
        "jobs": propagation_jobs,
    }
    return selections, levels, job_ids, contributing_counts, frequencies


def build_distributed_levels(normalised, core, *, wind_input_path: Path | None = None):
    """构造混合环境噪声谱。

    默认风噪声使用 Hildebrand 2021 接收谱；试验模式可改用东印度洋论文
    式(14)的非均匀风面源。降雨与 AIS 船舶使用空间源网格；低频传播使用
    RAM，高频使用 Bellhop。论文风、雨模型严格限制在 500–6400 Hz。
    """
    config = data_config(core)
    wind_model = str(normalised.get("windSourceModel", "hildebrand2021")).strip().lower()
    try:
        products = ensure_source_products(
            config,
            normalised.get("requestedTime"),
            wind_input_path=wind_input_path,
        )
        wind_path, wind_time_index, wind_time = products["wind"]
        rain_path, rain_time_index, rain_time = products["rain"]
        shipping_path, shipping_time_index, shipping_time = products["shipping"]
        if len({wind_time, rain_time, shipping_time}) != 1:
            raise ValueError(
                "风、降雨和 AIS 航运噪声源时次不一致："
                f"wind={wind_time}, rain={rain_time}, shipping={shipping_time}。"
            )
        wind_selection = load_local_sources(
            wind_path,
            wind_time_index,
            normalised["receiverLon"],
            normalised["receiverLat"],
            normalised["radiusKm"],
            int(config.get("MAX_WIND_SOURCE_CELLS", config.get("MAX_SOURCE_CELLS", 441))),
        )
        rain_selection = load_local_rain_sources(
            rain_path,
            rain_time_index,
            normalised["receiverLon"],
            normalised["receiverLat"],
            normalised["radiusKm"],
            int(config.get("MAX_RAIN_SOURCE_CELLS", config.get("MAX_SOURCE_CELLS", 441))),
        )
        shipping_selection = load_local_shipping_sources(
            shipping_path,
            shipping_time_index,
            normalised["receiverLon"],
            normalised["receiverLat"],
            normalised["radiusKm"],
            int(config.get("MAX_SHIPPING_SOURCE_CELLS", 5000)),
        )
        wind_sample = sample_wind_at_receiver(
            wind_path,
            wind_time_index,
            normalised["receiverLon"],
            normalised["receiverLat"],
        )
    except (OSError, ValueError, IndexError) as exc:
        raise core.PortalRequestError(str(exc)) from exc

    if wind_model == "liu2022_refined":
        wind_selection = uniform_polar_wind_selection(
            wind_selection,
            wind_sample.wind_speed_mps,
            sector_count=int(config.get("LIU2022_REFINED_SECTOR_COUNT", 24)),
            radial_step_km=float(config.get("LIU2022_REFINED_RADIAL_STEP_KM", 2.5)),
        )

    selections = {
        "wind": wind_selection,
        "rain": rain_selection,
        "shipping": shipping_selection,
    }

    base_anchors = propagation_frequencies(
        normalised["minFreq"],
        normalised["maxFreq"],
        int(config.get("PROPAGATION_FREQUENCY_COUNT", 5)),
    )
    # 63 Hz 是当前航运产品的第一个实测参考频点，500 Hz 是雨噪声模型及
    # 高低频传播模型的分界。把二者加入锚点，避免跨越物理边界做长距离插值。
    mandatory = np.asarray(
        [63.0, RAIN_MODEL_MIN_FREQUENCY_HZ, RAIN_MODEL_MAX_FREQUENCY_HZ],
        dtype=np.float64,
    )
    mandatory = mandatory[
        (mandatory >= float(normalised["minFreq"]))
        & (mandatory <= float(normalised["maxFreq"]))
    ]
    anchors = np.unique(np.concatenate((base_anchors, mandatory)))
    if wind_model == "liu2022_refined":
        paper_anchors = np.asarray(
            [500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
             2500.0, 3200.0, 4000.0, 5000.0, 6400.0],
            dtype=np.float64,
        )
        paper_anchors = paper_anchors[
            (paper_anchors >= float(normalised["minFreq"]))
            & (paper_anchors <= float(normalised["maxFreq"]))
        ]
        anchors = np.unique(np.concatenate((anchors, paper_anchors)))
    levels = {name: {} for name in selections}
    response_frequency = np.linspace(
        float(normalised["minFreq"]),
        float(normalised["maxFreq"]),
        int(normalised["numPoints"]),
        dtype=np.float64,
    )
    if wind_model == "hildebrand2021":
        wind_curve = hildebrand2021_psd_db(
            response_frequency,
            wind_sample.wind_speed_mps,
            float(normalised["receiverDepthM"]),
        )
        levels["wind"] = {
            float(frequency): float(level)
            for frequency, level in zip(response_frequency, wind_curve)
        }
        normalised["_windModel"] = {
            **model_metadata(wind_sample.wind_speed_mps, float(normalised["receiverDepthM"])),
            "u10Mps": round(wind_sample.u10_mps, 6),
            "v10Mps": round(wind_sample.v10_mps, 6),
            "interpolation": wind_sample.interpolation,
            "product": wind_sample.product_path.name,
            "time": wind_sample.time_utc,
        }
    elif wind_model in {"liu2022", "liu2022_refined"}:
        normalised["_windSurfaceModel"] = {
            "model": "Liu2022_EastIndianOcean_SLW",
            "calibration": "regional_published_unit_area_source_level",
            "equation": (
                "SLW=48.78-3.74log10((f/400)^2+1)"
                "+(11.85+U/2.06)log10(U/5.15)"
            ),
            "sourceDepthRule": "c/(4f)",
            "sourceDepthRangeM": [0.058594, 0.75],
            "referenceSoundSpeedMps": 1500.0,
            "frequencyHz": [500.0, 6400.0],
            "regionalCalibration": "East Indian Ocean",
            "reference": "Liu Yunfeng et al. (2022), equation (14)",
        }
        if wind_model == "liu2022_refined":
            normalised["_windSurfaceModel"].update({
                "trialMode": "refined_uniform_polar_surface",
                "sourceDistribution": "uniform_polar_annular_cells",
                "uniformWindSpeedMps": round(wind_sample.wind_speed_mps, 6),
                "sectorCount": int(config.get("LIU2022_REFINED_SECTOR_COUNT", 24)),
                "radialStepKm": float(config.get("LIU2022_REFINED_RADIAL_STEP_KM", 2.5)),
                "surfaceSourceCellCount": wind_selection.source_count,
            })
    else:
        raise core.PortalRequestError(f"不支持的风噪声源模型：{wind_model}。")
    job_ids = []
    contributing_counts = {name: {} for name in selections}
    crossover_hz = float(config.get("PROPAGATION_MODEL_CROSSOVER_HZ", 500.0))
    low_frequency_model = str(config.get("LOW_FREQUENCY_ACOUSTIC_MODEL", "ram")).strip().lower()
    high_frequency_model = str(config.get("HIGH_FREQUENCY_ACOUSTIC_MODEL", "bellhop")).strip().lower()
    propagation_mode = str(normalised.get("propagationModel", "hybrid")).strip().lower()
    if propagation_mode in {"ram", "bellhop"}:
        low_frequency_model = propagation_mode
        high_frequency_model = propagation_mode
    propagation_jobs = []
    for frequency_hz in anchors:
        surface_source_depths = [float(shipping_selection.source_depth_m)]
        paper_surface_frequency = (
            RAIN_MODEL_MIN_FREQUENCY_HZ
            <= float(frequency_hz)
            <= RAIN_MODEL_MAX_FREQUENCY_HZ
        )
        if paper_surface_frequency and (
            wind_model in {"liu2022", "liu2022_refined"}
            or rain_selection.source_count
        ):
            surface_source_depths.append(rain_surface_source_depth_m(float(frequency_hz)))
        minimum_source_depth = min(surface_source_depths)
        maximum_source_depth = max(surface_source_depths)
        propagation_config = dict(config)
        if wind_model == "liu2022_refined" and paper_surface_frequency:
            propagation_config["SECTOR_COUNT"] = int(
                config.get("LIU2022_REFINED_SECTOR_COUNT", 24)
            )
        request = propagation_request(
            normalised,
            wind_selection,
            frequency_hz,
            propagation_config,
        )
        params, _, _, _ = core._portal_3d_params(request)
        propagation_model = (
            low_frequency_model if float(frequency_hz) < crossover_hz else high_frequency_model
        )
        apply_fixed_acoustic_model(params, propagation_model)
        depth_count = int(config.get("TL_DEPTH_COUNT", 41))
        range_count = int(config.get("TL_RANGE_COUNT", 121))
        params.update({
            "freq_hz": float(frequency_hz),
            "source_depth_m": float(normalised["receiverDepthM"]),
            "receiver_depth_min_m": float(minimum_source_depth),
            "receiver_depth_max_m": max(
                float(maximum_source_depth) + 1.0,
                float(normalised["receiverDepthM"]),
            ),
            "receiver_depth_auto_max": False,
            "receiver_depth_auto_scope": "area",
            "horizontal_tl_depth_m": float(minimum_source_depth),
            "tl_n_rd": depth_count,
            "tl_n_range": range_count,
            "n_rd": depth_count,
            "n_range": range_count,
            "wind_noise_reciprocity": True,
            "actual_receiver": {
                "longitude": normalised["receiverLon"],
                "latitude": normalised["receiverLat"],
                "depth_m": normalised["receiverDepthM"],
            },
        })
        job_id, job_dir, _ = core._run_portal_job(params)
        tl_path = Path(job_dir) / "tl3d_depth_theta_range.nc"
        component_results = {
            "shipping": propagated_shipping_level_db(
                shipping_selection, float(frequency_hz), tl_path
            ),
        }
        if paper_surface_frequency and wind_model in {"liu2022", "liu2022_refined"}:
            component_results["wind"] = propagated_liu2022_wind_level_db(
                wind_selection, float(frequency_hz), tl_path
            )
        if paper_surface_frequency:
            component_results["rain"] = propagated_rain_level_db(
                rain_selection, float(frequency_hz), tl_path
            )
        job_ids.append(job_id)
        propagation_jobs.append({
            "jobId": job_id,
            "frequencyHz": round(float(frequency_hz), 6),
            "model": propagation_model,
        })
        for name, (level_db, contributing) in component_results.items():
            levels[name][float(frequency_hz)] = level_db
            contributing_counts[name][f"{float(frequency_hz):g}"] = contributing
    normalised["_hybridPropagation"] = {
        "model": (
            "hybrid_by_frequency"
            if propagation_mode == "hybrid"
            else f"{propagation_mode}_all_frequencies"
        ),
        "selectionMode": propagation_mode,
        "crossoverHz": crossover_hz,
        "lowFrequencyModel": low_frequency_model,
        "highFrequencyModel": high_frequency_model,
        "jobs": propagation_jobs,
    }
    return selections, levels, job_ids, contributing_counts, anchors


def build_source_grid_vertical_directivity(
    normalised: Mapping[str, Any],
    selections: Mapping[str, Any],
    core,
    *,
    ice_concentration: float = 0.0,
) -> dict[str, Any]:
    """按论文方法运行垂直阵 Bellhop 复声压任务并形成互谱指向性。"""
    config = data_config(core)
    ice_config = core.CONFIG.get("ENVIRONMENT_NOISE_DATA") or {}
    if not isinstance(ice_config, dict):
        raise ValueError("data_config.json 中 ENVIRONMENT_NOISE_DATA 必须是对象。")
    count = int(ice_config.get("VERTICAL_ARRAY_ELEMENT_COUNT", 32))
    spacing_m = float(ice_config.get("VERTICAL_ARRAY_SPACING_M", 20.0))
    if not 3 <= count <= 64:
        raise ValueError("VERTICAL_ARRAY_ELEMENT_COUNT 必须位于 3 到 64 之间。")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("VERTICAL_ARRAY_SPACING_M 必须是正有限数。")
    top_depth_m = float(normalised["receiverDepthM"])
    element_depths = top_depth_m + spacing_m * np.arange(count, dtype=np.float64)
    bottom_depth_m = float(normalised["profileDepthMaxM"])
    if float(element_depths[-1]) >= bottom_depth_m:
        raise ValueError(
            f"垂直阵末端深度 {element_depths[-1]:.3f} 米必须小于当前位置海深 "
            f"{bottom_depth_m:.3f} 米；请调整接收深度、阵元数或阵元间距。"
        )

    frequency_hz = float(normalised["calcFrequency"])
    scan_step_deg = float(ice_config.get("VERTICAL_ARRAY_SCAN_STEP_DEG", 1.0))
    if not np.isfinite(scan_step_deg) or not 0.1 <= scan_step_deg <= 10.0:
        raise ValueError("VERTICAL_ARRAY_SCAN_STEP_DEG 必须位于 0.1 到 10 度之间。")
    scan_angles = np.arange(-90.0, 90.0 + 0.5 * scan_step_deg, scan_step_deg)
    # 论文实测处理剔除首尾阵元；索引对外仍按 1 起始报告。
    active_indexes = np.arange(1, count - 1, dtype=int)

    source_depths = [
        float(selections["wind"].source_depth_m),
        float(selections["shipping"].source_depth_m),
    ]
    if RAIN_MODEL_MIN_FREQUENCY_HZ <= frequency_hz <= RAIN_MODEL_MAX_FREQUENCY_HZ:
        source_depths.append(rain_surface_source_depth_m(frequency_hz))
    receiver_depth_min = max(0.01, min(source_depths))
    receiver_depth_max = max(source_depths) + max(1.0, spacing_m * 0.05)
    propagation_config = dict(config)
    propagation_config["SECTOR_COUNT"] = int(
        ice_config.get("VERTICAL_ARRAY_SECTOR_COUNT", config.get("SECTOR_COUNT", 8))
    )

    def run_element(item):
        element_index, element_depth = item
        request = propagation_request(
            normalised,
            selections["wind"],
            frequency_hz,
            propagation_config,
        )
        params, _, _, _ = core._portal_3d_params(request)
        apply_fixed_acoustic_model(params, "bellhop")
        params.update({
            "freq_hz": frequency_hz,
            "source_depth_m": float(element_depth),
            "receiver_depth_min_m": receiver_depth_min,
            "receiver_depth_max_m": receiver_depth_max,
            "receiver_depth_auto_max": False,
            "receiver_depth_auto_scope": "area",
            "horizontal_tl_depth_m": receiver_depth_min,
            "tl_n_rd": max(8, int(config.get("TL_DEPTH_COUNT", 41))),
            "tl_n_range": max(16, int(config.get("TL_RANGE_COUNT", 121))),
            "n_rd": max(8, int(config.get("TL_DEPTH_COUNT", 41))),
            "n_range": max(16, int(config.get("TL_RANGE_COUNT", 121))),
            "wind_noise_reciprocity": True,
            "actual_receiver": {
                "longitude": normalised["receiverLon"],
                "latitude": normalised["receiverLat"],
                "depth_m": float(element_depth),
            },
        })
        job_id, job_dir, _ = core._run_portal_job(params)
        return int(element_index), str(job_id), Path(job_dir)

    workers = max(1, min(int(ice_config.get("VERTICAL_ARRAY_WORKERS", 2)), count))
    jobs = []
    if workers == 1:
        jobs = [run_element(item) for item in enumerate(element_depths)]
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="vertical-array-csd",
        ) as pool:
            futures = [pool.submit(run_element, item) for item in enumerate(element_depths)]
            try:
                jobs = [future.result() for future in as_completed(futures)]
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    jobs.sort(key=lambda item: item[0])
    result = source_grid_vertical_directivity(
        frequency_hz,
        selections,
        [item[2] for item in jobs],
        element_depths,
        scan_angles,
        active_element_indexes=active_indexes,
        sound_speed_mps=float(ice_config.get("VERTICAL_ARRAY_SOUND_SPEED_MPS", 1500.0)),
        ice_concentration=float(ice_concentration),
    )
    result["metadata"] = {
        "method": "vertical_array_cross_spectral_density_bellhop",
        "reference": "Shan et al., Acta Acustica 46(6), 2021, equations (6)-(9)",
        "frequencyHz": frequency_hz,
        "elementCount": count,
        "activeElementNumbers": (active_indexes + 1).tolist(),
        "elementSpacingM": spacing_m,
        "elementDepthM": np.round(element_depths, 6).tolist(),
        "scanStepDeg": scan_step_deg,
        "soundSpeedMps": float(ice_config.get("VERTICAL_ARRAY_SOUND_SPEED_MPS", 1500.0)),
        "bellhopJobIds": [item[1] for item in jobs],
        "sourceIndependence": "C=sum_j S_j h_j h_j^H",
    }
    return result


def interpolate_distributed_levels(levels: Mapping[str, Mapping[float, float]]):
    """返回一个可对任意频率数组求值的插值器（对数频率 / 线性谱级）。

    与 ``distributed_response`` 一致：在 log(f) 上线性插值，越界时钳制到
    最近锚点，避免把仅有 63 Hz / 1000 Hz 参考的航运谱不受控外推。
    """
    def evaluate(frequency: Any) -> dict[str, np.ndarray]:
        freq = np.asarray(frequency, dtype=np.float64)
        result = {}
        for name, source_levels in levels.items():
            anchors = np.asarray(
                sorted(float(value) for value in source_levels), dtype=np.float64
            )
            if anchors.size == 0:
                result[name] = np.full_like(freq, -300.0)
                continue
            values = np.asarray(
                [float(source_levels[float(value)]) for value in anchors],
                dtype=np.float64,
            )
            if anchors.size == 1:
                result[name] = np.full_like(freq, values[0])
            else:
                result[name] = np.interp(np.log(freq), np.log(anchors), values)
            if name == "rain":
                result[name] = np.where(
                    (freq >= RAIN_MODEL_MIN_FREQUENCY_HZ)
                    & (freq <= RAIN_MODEL_MAX_FREQUENCY_HZ),
                    result[name],
                    -300.0,
                )
        return result

    return evaluate
