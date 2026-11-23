"""端到端水声信号传输 WebSocket 服务。

链路与 HydrophoneSimulator 原型保持一致：

    发射信号 -> Bellhop/解析多途 -> 环境噪声 -> 水听器 -> 前置放大器 -> ADC

Bellhop 模式复用 ``services.channel_impulse_response`` 生成的原生 arrivals，
解析模式用于无 Bellhop 环境时的快速联调。最终结果会写入任务目录，并通过
WebSocket 返回适合浏览器绘图的降采样波形、时频图与链路指标。
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import threading
import wave
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import WebSocketDisconnect
from scipy.signal import spectrogram

from .services.channel_impulse_response import build_channel_impulse_response


ProgressCallback = Callable[[str, int, str], None]
MAX_SAMPLE_COUNT = 2_400_000
MAX_WAVEFORM_POINTS = 2400

HYDROPHONE_PRESETS: dict[str, dict[str, Any]] = {
    "BK-8103": {
        "label": "B&K 8103",
        "sensitivity_db": -211.0,
        "f_min_hz": 0.1,
        "f_max_hz": 180_000.0,
        "curve_frequency_hz": (
            0.1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000,
            5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000,
            50_000, 60_000, 70_000, 80_000, 90_000, 100_000,
        ),
        "curve_sensitivity_db": (
            -211.0, -211.0, -211.0, -211.0, -211.0, -211.0, -211.0,
            -211.0, -211.0, -211.0, -211.0, -211.0, -211.0, -211.0,
            -211.5, -211.0, -211.3, -212.0, -212.8, -213.2, -212.5,
            -212.0, -209.0, -220.0,
        ),
        "curve_quality": "厂家典型曲线图数字化近似（非单台校准）",
        "curve_source": "HydrophoneSimulator v0.2.0 / B&K 8103.csv",
    },
    "BK-8106": {
        "label": "B&K 8106（内置前放）",
        "sensitivity_db": -173.0,
        "f_min_hz": 3.0,
        "f_max_hz": 80_000.0,
        "curve_frequency_hz": (
            3, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000,
            10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 55_000,
            60_000, 65_000, 70_000, 75_000, 80_000,
        ),
        "curve_sensitivity_db": (
            -186.0, -181.0, -178.0, -176.0, -175.2, -175.0, -175.0,
            -175.0, -175.0, -175.0, -175.0, -175.5, -176.0, -176.0,
            -176.0, -174.5, -171.0, -170.0, -171.0, -175.0, -181.0,
            -186.0, -190.0,
        ),
        "curve_quality": "厂家典型曲线图数字化近似（非单台校准）",
        "curve_source": "HydrophoneSimulator v0.2.0 / B&K 8106.csv",
    },
    "RESON-TC4013": {
        "label": "Reson TC4013",
        "sensitivity_db": -211.0,
        "f_min_hz": 1.0,
        "f_max_hz": 170_000.0,
        "curve_frequency_hz": (
            1, 5_000, 6_000, 7_000, 8_000, 10_000, 12_000, 15_000,
            20_000, 25_000, 30_000, 40_000, 50_000, 60_000, 70_000,
            80_000, 90_000, 100_000, 110_000, 120_000, 130_000,
            140_000, 150_000, 160_000, 170_000,
        ),
        "curve_sensitivity_db": (
            -211.0, -213.0, -213.4, -212.8, -213.0, -213.0, -212.2,
            -211.8, -212.5, -213.0, -212.6, -213.0, -212.6, -211.7,
            -212.5, -213.0, -211.8, -211.5, -212.0, -211.5, -211.0,
            -210.5, -211.0, -213.0, -216.0,
        ),
        "curve_quality": "厂家典型曲线图数字化近似（非单台校准）",
        "curve_source": "HydrophoneSimulator v0.2.0 / Reson TC4013.csv",
    },
}


class SignalTransmissionError(ValueError):
    """可直接展示给页面的信号传输参数或计算错误。"""


def _progress(callback: ProgressCallback | None, phase: str, percent: int, message: str) -> None:
    if callback:
        callback(phase, int(percent), message)


def _number(
    data: Mapping[str, Any],
    names: Sequence[str],
    default: float,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value: Any = default
    for name in names:
        if name in data and data[name] not in (None, ""):
            value = data[name]
            break
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SignalTransmissionError(f"{label}必须是数值。") from exc
    if not math.isfinite(result):
        raise SignalTransmissionError(f"{label}必须是有限数值。")
    if minimum is not None and result < minimum:
        raise SignalTransmissionError(f"{label}不能小于 {minimum:g}。")
    if maximum is not None and result > maximum:
        raise SignalTransmissionError(f"{label}不能大于 {maximum:g}。")
    return result


def _coordinate(data: Mapping[str, Any], prefix: str) -> tuple[float, float]:
    lon = _number(data, (f"{prefix}_lon", f"{prefix}Lon"), 0.0, f"{prefix}经度", minimum=-180, maximum=180)
    lat = _number(data, (f"{prefix}_lat", f"{prefix}Lat"), 0.0, f"{prefix}纬度", minimum=-90, maximum=90)
    return lon, lat


def _normalize_frequencies(request: Mapping[str, Any], sample_rate: int) -> list[float]:
    raw = request.get("frequencies_hz", request.get("frequenciesHz", [100.0]))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = [raw]
    values: list[float] = []
    for value in raw:
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(frequency) and 0.0 < frequency < sample_rate / 2.0:
            values.append(frequency)
    values = list(dict.fromkeys(values))
    if not values:
        raise SignalTransmissionError("至少需要一个低于奈奎斯特频率的有效发射频率。")
    if len(values) > 32:
        raise SignalTransmissionError("一次仿真最多支持 32 个发射频率。")
    return values


def _normalize_source_band_levels(
    request: Mapping[str, Any], frequencies: Sequence[float]
) -> list[float] | None:
    raw = request.get("source_band_levels_db", request.get("sourceBandLevelsDb"))
    if raw in (None, ""):
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SignalTransmissionError("发射频带谱级必须是数组。")
    if len(raw) != len(frequencies):
        raise SignalTransmissionError("发射频带谱级数量必须与发射频率数量一致。")
    try:
        levels = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise SignalTransmissionError("发射频带谱级包含非数值数据。") from exc
    if not all(math.isfinite(value) for value in levels):
        raise SignalTransmissionError("发射频带谱级包含非有限数值。")
    return levels


def _normalize_noise_spectrum(value: Any, label: str) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, Mapping):
        raise SignalTransmissionError(f"{label}必须是噪声频谱对象。")
    raw_frequency = value.get("frequency_hz", value.get("frequencies_hz"))
    raw_psd = value.get("psd_db", value.get("spectrum_db"))
    if (
        not isinstance(raw_frequency, Sequence)
        or isinstance(raw_frequency, (str, bytes))
        or not isinstance(raw_psd, Sequence)
        or isinstance(raw_psd, (str, bytes))
    ):
        raise SignalTransmissionError(f"{label}缺少 frequency_hz 或 psd_db 数组。")
    if len(raw_frequency) != len(raw_psd) or not 2 <= len(raw_frequency) <= 2000:
        raise SignalTransmissionError(f"{label}的频率与谱级数组长度必须相同且位于 2—2000。")
    try:
        frequency = np.asarray(raw_frequency, dtype=np.float64)
        psd_db = np.asarray(raw_psd, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SignalTransmissionError(f"{label}包含非数值数据。") from exc
    if not (np.all(np.isfinite(frequency)) and np.all(np.isfinite(psd_db))):
        raise SignalTransmissionError(f"{label}包含非有限数值。")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise SignalTransmissionError(f"{label}频率必须严格递增且大于 0 Hz。")
    return {
        "frequency_hz": frequency.tolist(),
        "psd_db": psd_db.tolist(),
        "model": str(value.get("model") or "Wenz/天气环境噪声"),
        "query_time": str(value.get("query_time") or value.get("time") or ""),
        "lon": value.get("lon"),
        "lat": value.get("lat"),
    }


def _normalize_source_route(
    request: Mapping[str, Any],
    source_lon: float,
    source_lat: float,
    source_depth_m: float,
) -> list[dict[str, Any]]:
    raw_route = request.get("source_route", request.get("sourceRoute"))
    if raw_route in (None, ""):
        return [{
            "lon": source_lon,
            "lat": source_lat,
            "depth_m": source_depth_m,
            "distance_m": 0.0,
        }]
    if not isinstance(raw_route, Sequence) or isinstance(raw_route, (str, bytes)):
        raise SignalTransmissionError("发射航线必须是航迹点数组。")
    route: list[dict[str, float]] = []
    cumulative_distance = 0.0
    previous: tuple[float, float] | None = None
    for index, point in enumerate(raw_route):
        if not isinstance(point, Mapping):
            raise SignalTransmissionError(f"发射航线第 {index + 1} 个航迹点格式无效。")
        lon = _number(point, ("lon", "longitude", "source_lon"), math.nan, f"航迹点 {index + 1} 经度", minimum=-180, maximum=180)
        lat = _number(point, ("lat", "latitude", "source_lat"), math.nan, f"航迹点 {index + 1} 纬度", minimum=-90, maximum=90)
        depth = _number(point, ("depth_m", "depth", "source_depth_m"), source_depth_m, f"航迹点 {index + 1} 深度", minimum=0.01)
        if previous is not None:
            cumulative_distance += _distance_m(previous[0], previous[1], lon, lat)
        requested_distance = point.get("distance_m", point.get("distanceM"))
        if requested_distance not in (None, ""):
            try:
                candidate = float(requested_distance)
            except (TypeError, ValueError):
                candidate = math.nan
            if math.isfinite(candidate) and candidate >= cumulative_distance:
                cumulative_distance = candidate
        if not route or lon != route[-1]["lon"] or lat != route[-1]["lat"]:
            route.append({
                "lon": lon,
                "lat": lat,
                "depth_m": depth,
                "distance_m": cumulative_distance,
                "noise_spectrum": _normalize_noise_spectrum(
                    point.get("noise_spectrum", point.get("noiseSpectrum")),
                    f"航迹点 {index + 1} 环境噪声谱",
                ),
            })
        previous = (lon, lat)
    if len(route) < 1:
        raise SignalTransmissionError("发射航线至少需要一个航迹点。")
    # if len(route) > 1000:
    #     raise SignalTransmissionError("一次传输计算最多支持1000个航迹点。")
    return route


def _normalize_request(request: Mapping[str, Any]) -> dict[str, Any]:
    sample_rate = int(round(_number(request, ("sample_rate", "sampleRate"), 48_000, "采样率", minimum=2000, maximum=192_000)))
    duration = _number(request, ("duration_s", "durationS"), 10.0, "仿真时长", minimum=0.05, maximum=3600)
    if int(round(sample_rate * duration)) > MAX_SAMPLE_COUNT:
        maximum_duration = MAX_SAMPLE_COUNT / sample_rate
        raise SignalTransmissionError(
            f"当前采样率下仿真时长不能超过 {maximum_duration:.1f} s（最多 {MAX_SAMPLE_COUNT:,} 个采样点）。"
        )
    source_lon, source_lat = _coordinate(request, "source")
    receiver_lon, receiver_lat = _coordinate(request, "receiver")
    if source_lon == receiver_lon and source_lat == receiver_lat:
        raise SignalTransmissionError("发射位置和接收位置不能重合。")
    hydrophone = str(request.get("hydrophone", "BK-8103")).strip().upper()
    if hydrophone not in HYDROPHONE_PRESETS:
        raise SignalTransmissionError(f"不支持的水听器型号：{hydrophone}")
    signal_type = str(request.get("signal_type", request.get("signalType", "CW"))).strip().upper()
    if signal_type not in {"CW", "TONE_BURST", "LFM"}:
        raise SignalTransmissionError(f"不支持的发射信号类型：{signal_type}")
    propagation_model = str(request.get("propagation_model", "bellhop")).strip().lower()
    if propagation_model not in {"bellhop", "analytic"}:
        raise SignalTransmissionError("传播模型只能是 bellhop 或 analytic。")
    bits = int(round(_number(request, ("adc_bits", "adcBits"), 24, "ADC 位数")))
    if bits not in {16, 24, 32}:
        raise SignalTransmissionError("ADC 位数仅支持 16、24、32 bit。")
    source_depth = _number(request, ("source_depth_m", "sourceDepth"), 50.0, "发射深度", minimum=0.01)
    frequencies = _normalize_frequencies(request, sample_rate)
    noise_model = str(request.get("noise_model", "white")).strip().lower()
    if noise_model not in {"white", "position_spectrum", "wenz"}:
        raise SignalTransmissionError(
            "环境噪声模型只能是 white、position_spectrum 或 wenz。"
        )
    normalized = {
        "request_id": str(request.get("request_id") or secrets.token_hex(6)),
        "sample_rate": sample_rate,
        "duration_s": duration,
        "signal_type": signal_type,
        "frequencies_hz": frequencies,
        "source_band_levels_db": _normalize_source_band_levels(request, frequencies),
        "pulse_width_ms": _number(request, ("pulse_width_ms", "pulseWidthMs"), 100.0, "脉宽", minimum=0.1),
        "pulse_repetition_s": _number(request, ("pulse_repetition_s", "pulseRepetitionS"), 1.0, "脉冲重复周期", minimum=0.001),
        "source_level_db": _number(request, ("source_level_db", "sourceLevelDb"), 180.0, "声源级"),
        "source_lon": source_lon,
        "source_lat": source_lat,
        "source_depth_m": source_depth,
        "receiver_lon": receiver_lon,
        "receiver_lat": receiver_lat,
        "receiver_depth_m": _number(request, ("receiver_depth_m", "receiverDepth"), 100.0, "接收深度", minimum=0.01),
        "water_depth_m": _number(request, ("water_depth_m", "waterDepth"), 300.0, "水深", minimum=1.0),
        "sound_speed_m_s": _number(request, ("sound_speed_m_s", "soundSpeed"), 1500.0, "声速", minimum=1000, maximum=2000),
        "depression_angle_deg": abs(_number(request, ("depression_angle_deg", "depressionAngleDeg"), 45.0, "俯角", maximum=90)),
        "elevation_angle_deg": abs(_number(request, ("elevation_angle_deg", "elevationAngleDeg"), 45.0, "仰角", maximum=90)),
        "arrival_beam_count": int(round(_number(request, ("arrival_beam_count", "arrivalBeamCount"), 2001, "Bellhop 波束数", minimum=1, maximum=100_001))),
        "propagation_model": propagation_model,
        "surface_echo": bool(request.get("surface_echo", True)),
        "bottom_echo": bool(request.get("bottom_echo", True)),
        "noise_enabled": bool(request.get("noise_enabled", True)),
        "noise_level_db": _number(request, ("noise_level_db", "noiseLevelDb"), 20.0, "噪声谱级"),
        "noise_model": noise_model,
        "receiver_noise_spectrum": _normalize_noise_spectrum(
            request.get("receiver_noise_spectrum", request.get("receiverNoiseSpectrum")),
            "接收位置环境噪声谱",
        ),
        "hydrophone": hydrophone,
        "sensitivity_db": request.get("sensitivity_db"),
        # B&K 8106 的厂家灵敏度曲线已经包含内置前放。这里的增益始终表示
        # 曲线之后的附加外置前放，因此 8106 未显式配置时默认为 0 dB。
        "preamp_gain_db": _number(
            request,
            ("preamp_gain_db", "preampGainDb"),
            0.0 if hydrophone == "BK-8106" else 6.0,
            "附加外部前置放大器增益" if hydrophone == "BK-8106" else "前置放大器增益",
        ),
        "adc_bits": bits,
        "adc_full_scale_v": _number(request, ("adc_full_scale_v", "adcFullScaleV"), 2.0, "ADC 满量程", minimum=1e-9),
        "random_seed": int(round(_number(request, ("random_seed", "randomSeed"), 20261109, "随机种子"))),
        "dataSource": request.get("dataSource", "EnvReanalysis"),
        "dataType": request.get("dataType", "DT_ENV_SOUND"),
        "time": request.get("time", "2025-01-04 12:00:00"),
        "timeRes": request.get("timeRes", 0),
    }
    if normalized["source_depth_m"] >= normalized["water_depth_m"]:
        raise SignalTransmissionError("发射深度必须小于水深。")
    if normalized["receiver_depth_m"] >= normalized["water_depth_m"]:
        raise SignalTransmissionError("接收深度必须小于水深。")
    if normalized["depression_angle_deg"] == 0 and normalized["elevation_angle_deg"] == 0:
        raise SignalTransmissionError("俯角和仰角不能同时为 0。")
    normalized["source_route"] = _normalize_source_route(
        request, source_lon, source_lat, source_depth
    )
    for index, point in enumerate(normalized["source_route"]):
        if point["depth_m"] >= normalized["water_depth_m"]:
            raise SignalTransmissionError(f"航迹站位 {index + 1} 的发射深度必须小于水深。")
    if normalized["noise_enabled"] and normalized["noise_model"] == "position_spectrum":
        if normalized["receiver_noise_spectrum"] is None:
            raise SignalTransmissionError("缺少固定接收位置的环境噪声频谱。")
        missing = [
            str(index + 1) for index, point in enumerate(normalized["source_route"])
            if point.get("noise_spectrum") is None
        ]
        if missing:
            raise SignalTransmissionError(f"航迹站位 {', '.join(missing)} 缺少对应位置的环境噪声频谱。")
    return normalized


def _parse_utc_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _wenz_spectrum_from_forecast(
    forecast: Mapping[str, Any],
    *,
    requested_time: Any,
    longitude: float,
    latitude: float,
) -> dict[str, Any]:
    """Select one forecast frame and expose it in the chain noise-spectrum shape."""
    frequency = forecast.get("frequency_hz")
    frames = forecast.get("frames")
    if not isinstance(frequency, Sequence) or isinstance(frequency, (str, bytes)):
        raise SignalTransmissionError("Wenz 环境噪声结果缺少频率轴。")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        raise SignalTransmissionError("Wenz 环境噪声结果缺少预报帧。")

    requested = _parse_utc_time(requested_time)
    selected_index = int((forecast.get("slider") or {}).get("initial", 0))
    if requested is not None:
        candidates: list[tuple[float, int]] = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                continue
            frame_time = _parse_utc_time(frame.get("time"))
            if frame_time is not None:
                candidates.append((abs((frame_time - requested).total_seconds()), index))
        if candidates:
            selected_index = min(candidates)[1]
    selected_index = max(0, min(selected_index, len(frames) - 1))
    frame = frames[selected_index]
    if not isinstance(frame, Mapping):
        raise SignalTransmissionError("Wenz 环境噪声预报帧格式无效。")
    varying = frame.get("varying_components_db")
    psd_db = varying.get("total") if isinstance(varying, Mapping) else None
    if not isinstance(psd_db, Sequence) or isinstance(psd_db, (str, bytes)):
        raise SignalTransmissionError("Wenz 环境噪声预报帧缺少总噪声谱。")
    if len(frequency) != len(psd_db):
        raise SignalTransmissionError("Wenz 环境噪声频率轴与总噪声谱长度不一致。")
    return {
        "frequency_hz": list(frequency),
        "psd_db": list(psd_db),
        "model": str(forecast.get("model") or "Wenz/天气环境噪声"),
        "query_time": str(frame.get("time") or ""),
        "lon": float(longitude),
        "lat": float(latitude),
        "forecast_frame_index": selected_index,
        "drivers": dict(frame.get("drivers") or {}),
        "weather": dict(frame.get("weather") or {}),
        "forecast_provider": (forecast.get("forecast") or {}).get("provider"),
    }


def _fetch_wenz_ais(config: Mapping[str, Any], core: Any) -> dict[str, Any]:
    """Reuse the backend rolling AIS feed; the browser does not build noise spectra."""
    from .api_endpoints import get_realtime_ais as ais_endpoint

    longitudes = [float(config["receiver_lon"])] + [
        float(point["lon"]) for point in config["source_route"]
    ]
    latitudes = [float(config["receiver_lat"])] + [
        float(point["lat"]) for point in config["source_route"]
    ]
    margin_deg = 5.0
    west = max(-180.0, min(longitudes) - margin_deg)
    east = min(180.0, max(longitudes) + margin_deg)
    south = max(-90.0, min(latitudes) - margin_deg)
    north = min(90.0, max(latitudes) + margin_deg)
    if west >= east:
        west, east = -180.0, 180.0
    if south >= north:
        south, north = -90.0, 90.0
    return ais_endpoint.calculate({
        "bottomLeftX": west,
        "bottomLeftY": south,
        "topRightX": east,
        "topRightY": north,
        "durationSeconds": 300,
    }, core)


def _fetch_wenz_forecast(
    longitude: float,
    latitude: float,
    sample_rate: int,
    ais: Mapping[str, Any],
    core: Any,
) -> dict[str, Any]:
    from .api_endpoints import analyse_forecast_environment_noise as noise_endpoint

    return noise_endpoint.calculate({
        "coordX": longitude,
        "coordY": latitude,
        "forecastHours": 48,
        "minFreq": 10.0,
        "maxFreq": min(100_000.0, sample_rate / 2.0),
        "numPoints": 512,
        "ais": ais,
    }, core)


def _resolve_wenz_noise_spectra(
    config: dict[str, Any],
    core: Any,
    progress: ProgressCallback | None = None,
) -> None:
    if not config["noise_enabled"] or config["noise_model"] != "wenz":
        return
    _progress(progress, "noise", 3, "正在获取 AIS 与气象并生成 Wenz 环境噪声谱")
    try:
        ais = _fetch_wenz_ais(config, core)
        cache: dict[tuple[float, float], dict[str, Any]] = {}

        def spectrum_at(longitude: float, latitude: float) -> dict[str, Any]:
            key = (round(float(longitude), 6), round(float(latitude), 6))
            if key not in cache:
                forecast = _fetch_wenz_forecast(
                    float(longitude), float(latitude), int(config["sample_rate"]), ais, core
                )
                cache[key] = _wenz_spectrum_from_forecast(
                    forecast,
                    requested_time=config.get("time"),
                    longitude=float(longitude),
                    latitude=float(latitude),
                )
            return cache[key]

        config["receiver_noise_spectrum"] = spectrum_at(
            float(config["receiver_lon"]), float(config["receiver_lat"])
        )
        route = config["source_route"]
        total = len(route) + 1
        step = max(1, total // 50)
        for index, point in enumerate(route, 2):
            point["noise_spectrum"] = spectrum_at(float(point["lon"]), float(point["lat"]))
            if index == 2 or index % step == 0 or index == total:
                _progress(
                    progress, "noise", min(10, 3 + 7 * index // total),
                    f"正在生成 Wenz 环境噪声谱 {index}/{total}",
                )
        config["wenz_context"] = {
            "ais_source": ais.get("source"),
            "ais_snapshot_time_utc": ais.get("snapshot_time_utc"),
            "ais_vessel_count": ais.get("vessel_count", len(ais.get("vessels") or [])),
            "history_window_seconds": ais.get("history_window_seconds"),
        }
    except SignalTransmissionError:
        raise
    except Exception as exc:
        raise SignalTransmissionError(f"Wenz 环境噪声生成失败：{exc}") from exc


def _source_signal(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = int(config["sample_rate"])
    count = int(round(sample_rate * float(config["duration_s"])))
    time_s = np.arange(count, dtype=np.float64) / sample_rate
    frequencies = np.asarray(config["frequencies_hz"], dtype=np.float64)
    total_rms_pa = 1e-6 * 10.0 ** (float(config["source_level_db"]) / 20.0)
    raw_band_levels = config.get("source_band_levels_db")
    if raw_band_levels is not None:
        band_levels = np.asarray(raw_band_levels, dtype=np.float64)
        relative_rms = np.power(10.0, (band_levels - float(np.max(band_levels))) / 20.0)
        component_rms = total_rms_pa * relative_rms / math.sqrt(float(np.sum(relative_rms**2)))
    else:
        component_rms = np.full(len(frequencies), total_rms_pa / math.sqrt(len(frequencies)))
    component_peaks = math.sqrt(2.0) * component_rms
    signal_type = str(config["signal_type"])
    if signal_type == "LFM":
        width_s = max(1.0 / sample_rate, float(config["pulse_width_ms"]) / 1000.0)
        repetition_s = max(width_s, float(config["pulse_repetition_s"]))
        local_time = np.mod(time_s, repetition_s)
        active = local_time < width_s
        start = float(np.min(frequencies))
        stop = float(np.max(frequencies)) if len(frequencies) > 1 else min(sample_rate / 2.1, start * 2.0)
        sweep_rate = (stop - start) / width_s
        phase = 2.0 * np.pi * (start * local_time + 0.5 * sweep_rate * local_time**2)
        envelope = np.sin(np.pi * np.clip(local_time / width_s, 0.0, 1.0)) ** 2
        source = math.sqrt(2.0) * total_rms_pa * np.sin(phase) * envelope * active
    else:
        source = np.zeros(count, dtype=np.float64)
        for frequency, peak in zip(frequencies, component_peaks, strict=True):
            source += peak * np.sin(2.0 * np.pi * frequency * time_s)
        if signal_type == "TONE_BURST":
            width_s = max(1.0 / sample_rate, float(config["pulse_width_ms"]) / 1000.0)
            repetition_s = max(width_s, float(config["pulse_repetition_s"]))
            local_time = np.mod(time_s, repetition_s)
            active = local_time < width_s
            envelope = np.sin(np.pi * np.clip(local_time / width_s, 0.0, 1.0)) ** 2
            source *= envelope * active
    return time_s, np.asarray(source, dtype=np.float64)


def _distance_m(source_lon: float, source_lat: float, receiver_lon: float, receiver_lat: float) -> float:
    lat1, lat2 = math.radians(source_lat), math.radians(receiver_lat)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(receiver_lon - source_lon)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 6_371_008.8 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


def _thorp_db_per_km(frequency_hz: float) -> float:
    f = max(float(frequency_hz) / 1000.0, 1e-6)
    f2 = f * f
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def _analytic_multipath(config: Mapping[str, Any]) -> dict[str, Any]:
    horizontal = _distance_m(
        float(config["source_lon"]), float(config["source_lat"]),
        float(config["receiver_lon"]), float(config["receiver_lat"]),
    )
    source_depth = float(config["source_depth_m"])
    receiver_depth = float(config["receiver_depth_m"])
    water_depth = float(config["water_depth_m"])
    sound_speed = float(config["sound_speed_m_s"])
    frequency = float(np.mean(config["frequencies_hz"]))
    alpha = _thorp_db_per_km(frequency)
    geometries: list[tuple[str, float, float]] = [
        ("direct", math.hypot(horizontal, receiver_depth - source_depth), 1.0),
    ]
    if config["surface_echo"]:
        geometries.append(("surface", math.hypot(horizontal, receiver_depth + source_depth), -1.0))
    if config["bottom_echo"]:
        image_depth = 2.0 * water_depth - source_depth
        geometries.append(("bottom", math.hypot(horizontal, image_depth - receiver_depth), 0.6))
    taps: list[dict[str, Any]] = []
    for index, (kind, path_m, reflection) in enumerate(geometries, 1):
        tl_db = 20.0 * math.log10(max(path_m, 1.0)) + alpha * path_m / 1000.0
        amplitude = abs(reflection) * 10.0 ** (-tl_db / 20.0)
        phase = math.pi if reflection < 0 else 0.0
        taps.append({
            "arrival_index": index,
            "path_type": kind,
            "delay_s": path_m / sound_speed,
            "amplitude": amplitude,
            "phase_deg": math.degrees(phase),
            "gain_real": amplitude * math.cos(phase),
            "gain_imag": amplitude * math.sin(phase),
            "source_angle_deg": None,
            "arrival_angle_deg": None,
            "top_bounces": 1 if kind == "surface" else 0,
            "bottom_bounces": 1 if kind == "bottom" else 0,
        })
    taps.sort(key=lambda item: item["delay_s"])
    return {
        "job_id": secrets.token_hex(6),
        "engine": "解析三路径",
        "distance_m": horizontal,
        "frequency_hz": frequency,
        "path_count": len(taps),
        "first_arrival_s": taps[0]["delay_s"],
        "impulse_response": taps,
    }


def _bellhop_multipath(config: Mapping[str, Any], core: Any) -> dict[str, Any]:
    payload = {
        'allowEmptyArrivals': bool(config.get('allow_empty_arrivals', False)),
        "sourceLon": config["source_lon"],
        "sourceLat": config["source_lat"],
        "sourceDepth": config["source_depth_m"],
        "receiverLon": config["receiver_lon"],
        "receiverLat": config["receiver_lat"],
        "receiverDepth": config["receiver_depth_m"],
        "depressionAngleDeg": config["depression_angle_deg"],
        "elevationAngleDeg": config["elevation_angle_deg"],
        "frequencyHz": float(np.mean(config["frequencies_hz"])),
        "arrivalBeamCount": config["arrival_beam_count"],
        "dataSource": config["dataSource"],
        "dataType": config["dataType"],
        "time": config["time"],
        "timeRes": config["timeRes"],
    }
    response, _job_id = build_channel_impulse_response(payload, core)
    return response


def _apply_multipath(source: np.ndarray, sample_rate: int, taps: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not taps:
        raise SignalTransmissionError("传播模型没有返回任何有效声到达。")
    frequency_hz = np.fft.rfftfreq(len(source), 1.0 / sample_rate)
    first_delay = min(float(tap["delay_s"]) for tap in taps)
    response = np.zeros_like(frequency_hz, dtype=np.complex128)
    for tap in taps:
        gain = complex(float(tap.get("gain_real", 0.0)), float(tap.get("gain_imag", 0.0)))
        relative_delay = max(0.0, float(tap["delay_s"]) - first_delay)
        response += gain * np.exp(-2j * np.pi * frequency_hz * relative_delay)
    return np.fft.irfft(np.fft.rfft(source) * response, n=len(source))


def _spectrum_noise(
    sample_count: int,
    sample_rate: int,
    spectrum: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    frequencies = np.asarray(spectrum["frequency_hz"], dtype=np.float64)
    psd_db = np.asarray(spectrum["psd_db"], dtype=np.float64)
    fft_frequency = np.fft.rfftfreq(sample_count, 1.0 / sample_rate)
    log_source = np.log(np.maximum(frequencies, 1e-9))
    log_target = np.log(np.maximum(fft_frequency, frequencies[0]))
    interpolated_db = np.interp(log_target, log_source, psd_db, left=psd_db[0], right=psd_db[-1])
    psd_pa2_hz = 1e-12 * np.power(10.0, interpolated_db / 10.0)
    random = np.random.default_rng(seed)
    white = random.normal(0.0, 1.0, sample_count)
    shaped = np.fft.irfft(np.fft.rfft(white) * np.sqrt(psd_pa2_hz), n=sample_count)
    rms = float(np.sqrt(np.mean(shaped ** 2)))
    target_rms = math.sqrt(max(float(np.trapezoid(psd_pa2_hz, fft_frequency)), 1e-30))
    return shaped * (target_rms / max(rms, 1e-30))


def _add_noise(
    signal: np.ndarray,
    config: Mapping[str, Any],
    spectrum: Mapping[str, Any] | None = None,
    *,
    seed_offset: int = 0,
) -> np.ndarray:
    if not config["noise_enabled"]:
        return signal.copy()
    sample_rate = int(config["sample_rate"])
    if spectrum is not None:
        noise = _spectrum_noise(
            len(signal), sample_rate, spectrum,
            int(config["random_seed"]) + int(seed_offset),
        )
        return signal + noise
    psd_pa2_hz = 1e-12 * 10.0 ** (float(config["noise_level_db"]) / 10.0)
    rms_pa = math.sqrt(psd_pa2_hz * sample_rate / 2.0)
    random = np.random.default_rng(int(config["random_seed"]) + int(seed_offset))
    return signal + random.normal(0.0, rms_pa, size=len(signal))


def _hydrophone_sensitivity_response_db(
    frequencies_hz: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Return the selected hydrophone sensitivity on logarithmic frequency axes."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    preset = HYDROPHONE_PRESETS[str(config["hydrophone"])]
    sensitivity = config.get("sensitivity_db")
    band = (frequencies >= float(preset["f_min_hz"])) & (frequencies <= float(preset["f_max_hz"]))
    response_db = np.full_like(frequencies, -np.inf)
    if sensitivity not in (None, ""):
        response_db[band] = float(sensitivity)
        return response_db

    curve_frequency = np.asarray(preset["curve_frequency_hz"], dtype=np.float64)
    curve_sensitivity = np.asarray(preset["curve_sensitivity_db"], dtype=np.float64)
    response_db[band] = np.interp(
        np.log10(frequencies[band]),
        np.log10(curve_frequency),
        curve_sensitivity,
        left=float(curve_sensitivity[0]),
        right=float(curve_sensitivity[-1]),
    )
    return response_db


def _hydrophone_chain(received_pa: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sample_rate = int(config["sample_rate"])
    frequencies = np.fft.rfftfreq(len(received_pa), 1.0 / sample_rate)
    sensitivity_db = _hydrophone_sensitivity_response_db(frequencies, config)
    response = np.zeros_like(frequencies)
    finite = np.isfinite(sensitivity_db)
    response[finite] = 10.0 ** (sensitivity_db[finite] / 20.0)
    gain = 10.0 ** (float(config["preamp_gain_db"]) / 20.0)
    voltage = np.fft.irfft(np.fft.rfft(received_pa) * (1e6 * response * gain), n=len(received_pa))
    full_scale = float(config["adc_full_scale_v"])
    clipped = np.clip(voltage, -full_scale, full_scale)
    max_code = (1 << (int(config["adc_bits"]) - 1)) - 1
    adc_codes = np.rint(clipped / full_scale * max_code).astype(np.int64)
    adc_normalized = (adc_codes / max_code).astype(np.float32)
    clipping_percent = float(np.mean(np.abs(voltage) > full_scale) * 100.0)
    return voltage, adc_codes, adc_normalized, clipping_percent


def _adc_calibrated_pressure_pa(
    adc_normalized: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Invert ADC scaling, external gain and hydrophone sensitivity to pressure."""
    samples = np.asarray(adc_normalized, dtype=np.float64)
    if not len(samples):
        return samples
    sample_rate = int(config["sample_rate"])
    frequencies = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    sensitivity_db = _hydrophone_sensitivity_response_db(frequencies, config)
    sensitivity_v_per_upa = np.zeros_like(frequencies)
    finite = np.isfinite(sensitivity_db)
    sensitivity_v_per_upa[finite] = 10.0 ** (sensitivity_db[finite] / 20.0)
    external_gain = 10.0 ** (float(config["preamp_gain_db"]) / 20.0)
    denominator = 1e6 * sensitivity_v_per_upa * external_gain
    voltage = samples * float(config["adc_full_scale_v"])
    voltage_spectrum = np.fft.rfft(voltage)
    pressure_spectrum = np.zeros_like(voltage_spectrum)
    valid = denominator > 0.0
    pressure_spectrum[valid] = voltage_spectrum[valid] / denominator[valid]
    return np.fft.irfft(pressure_spectrum, n=len(samples))


def _decimate_waveform(time_s: np.ndarray, values: np.ndarray, limit: int = MAX_WAVEFORM_POINTS) -> dict[str, list[float]]:
    if len(values) <= limit:
        indices = np.arange(len(values))
    else:
        indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return {
        "time_s": np.round(time_s[indices], 7).tolist(),
        "values": np.round(values[indices], 9).tolist(),
    }


def _spectrogram_payload(values: np.ndarray, sample_rate: int, unit: str = "dBFS/Hz") -> dict[str, Any]:
    segment = min(1024, len(values))
    if segment < 32:
        return {"time_s": [], "frequency_hz": [], "psd_db": [], "dbfs_per_hz": [], "unit": unit}
    frequency, time_s, power = spectrogram(
        values, fs=sample_rate, nperseg=segment, noverlap=segment // 2,
        scaling="density", mode="psd",
    )
    frequency_indices = np.linspace(0, len(frequency) - 1, min(128, len(frequency)), dtype=int)
    time_indices = np.linspace(0, len(time_s) - 1, min(200, len(time_s)), dtype=int)
    db = 10.0 * np.log10(np.maximum(power[np.ix_(frequency_indices, time_indices)], 1e-20))
    return {
        "time_s": np.round(time_s[time_indices], 5).tolist(),
        "frequency_hz": np.round(frequency[frequency_indices], 3).tolist(),
        "psd_db": np.round(db, 3).tolist(),
        "dbfs_per_hz": np.round(db, 3).tolist(),
        "unit": unit,
    }


def _write_result_files(
    jobs_dir: str | Path,
    job_id: str,
    sample_rate: int,
    adc_codes: np.ndarray,
    adc_bits: int,
    adc_normalized: np.ndarray,
    source_monitor: np.ndarray,
    result: Mapping[str, Any],
) -> dict[str, str]:
    output_dir = Path(jobs_dir) / job_id / "signal_transmission"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "hydrophone_adc.wav"
    if adc_bits == 16:
        pcm = np.asarray(adc_codes, dtype="<i2").tobytes()
        sample_width = 2
    elif adc_bits == 24:
        integers = np.asarray(adc_codes, dtype="<i4")
        pcm = integers.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        sample_width = 3
    else:
        pcm = np.asarray(adc_codes, dtype="<i4").tobytes()
        sample_width = 4
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    receiver_wav_path = output_dir / "receiver_monitor.wav"
    receiver_pcm = np.rint(np.clip(adc_normalized, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(receiver_wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(receiver_pcm.tobytes())
    source_wav_path = output_dir / "source_route_monitor.wav"
    source_peak = float(np.max(np.abs(source_monitor))) if len(source_monitor) else 0.0
    source_pcm = np.rint(
        np.clip(source_monitor / max(source_peak, 1e-30), -1.0, 1.0) * 32767.0
    ).astype("<i2")
    with wave.open(str(source_wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(source_pcm.tobytes())
    json_path = output_dir / "signal_transmission_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "adc_wav": f"/file/{job_id}/signal_transmission/{wav_path.name}",
        "receiver_monitor_wav": f"/file/{job_id}/signal_transmission/{receiver_wav_path.name}",
        "source_monitor_wav": f"/file/{job_id}/signal_transmission/{source_wav_path.name}",
        "result_json": f"/file/{job_id}/signal_transmission/{json_path.name}",
    }


def simulate_signal_transmission(
    request: Mapping[str, Any],
    *,
    jobs_dir: str | Path,
    core: Any,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """同步执行完整信号链；由 WebSocket 处理器放入工作线程。"""
    config = _normalize_request(request)
    _resolve_wenz_noise_spectra(config, core, progress)
    _progress(progress, "transmit", 8, "正在生成发射信号")
    time_s, source = _source_signal(config)

    route = list(config["source_route"])
    propagated = np.zeros_like(source)
    source_monitor = source.copy()
    route_results: list[dict[str, Any]] = []
    distances = np.asarray([float(point["distance_m"]) for point in route], dtype=np.float64)
    if len(route) == 1 or not (distances[-1] > 0.0):
        boundaries = np.linspace(0, len(source), len(route) + 1, dtype=int)
    else:
        normalized_distance = distances / distances[-1]
        midpoints = 0.5 * (normalized_distance[:-1] + normalized_distance[1:])
        boundaries = np.rint(
            np.concatenate(([0.0], midpoints, [1.0])) * len(source)
        ).astype(int)

    for index, point in enumerate(route):
        percent = 18 + round(44 * index / max(1, len(route)))
        _progress(
            progress,
            "propagate",
            percent,
            f"正在计算航迹站位 {index + 1}/{len(route)} 的传播与多途声到达",
        )
        station_config = dict(config)
        station_config.update({
            "source_lon": point["lon"],
            "source_lat": point["lat"],
            "source_depth_m": point["depth_m"],
        })
        try:
            if config["propagation_model"] == "bellhop":
                station_propagation = _bellhop_multipath(station_config, core)
            else:
                station_propagation = _analytic_multipath(station_config)
        except Exception as exc:
            raise SignalTransmissionError(
                f"航迹站位 {index + 1}/{len(route)}（{point['lon']:.6f}, "
                f"{point['lat']:.6f}）传播计算失败：{exc}"
            ) from exc
        taps = station_propagation.get("impulse_response") or []
        station_signal = _apply_multipath(source, int(config["sample_rate"]), taps)
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        propagated[start:stop] = station_signal[start:stop]
        if config["noise_enabled"] and stop > start:
            source_monitor[start:stop] = _add_noise(
                source[start:stop],
                config,
                point.get("noise_spectrum"),
                seed_offset=1000 + index,
            )
        route_results.append({
            "station_index": index + 1,
            "source": {
                "lon": point["lon"],
                "lat": point["lat"],
                "depth_m": point["depth_m"],
                "route_distance_m": point["distance_m"],
                "noise_model": (point.get("noise_spectrum") or {}).get("model"),
                "noise_query_time": (point.get("noise_spectrum") or {}).get("query_time"),
            },
            "job_id": station_propagation.get("job_id"),
            "engine": station_propagation.get("engine"),
            "distance_m": station_propagation.get("distance_m"),
            "bearing_deg": station_propagation.get("bearing_deg"),
            "frequency_hz": station_propagation.get("frequency_hz"),
            "path_count": len(taps),
            "first_arrival_s": station_propagation.get("first_arrival_s"),
            "impulse_response": taps,
            "sample_start": start,
            "sample_end": stop,
        })

    propagation = route_results[0]
    taps = propagation["impulse_response"]

    _progress(progress, "noise", 62, "正在叠加声源航迹与固定接收位置的环境噪声频谱")
    received = _add_noise(
        propagated,
        config,
        config.get("receiver_noise_spectrum"),
        seed_offset=10_000,
    )

    _progress(progress, "receive", 74, "正在计算水听器、前放与 ADC")
    voltage, adc_codes, adc_normalized, clipping_percent = _hydrophone_chain(received, config)
    adc_calibrated_pressure = _adc_calibrated_pressure_pa(adc_normalized, config)

    _progress(progress, "analyse", 86, "正在整理波形、时频图与链路指标")
    source_rms = float(np.sqrt(np.mean(source**2)))
    received_rms = float(np.sqrt(np.mean(received**2)))
    signal_rms = float(np.sqrt(np.mean(propagated**2)))
    noise_rms = float(np.sqrt(np.mean((received - propagated) ** 2)))
    metrics = {
        "source_pressure_rms_pa": source_rms,
        "received_pressure_rms_upa": received_rms * 1e6,
        "received_signal_rms_upa": signal_rms * 1e6,
        "estimated_snr_db": 20.0 * math.log10(max(signal_rms, 1e-30) / max(noise_rms, 1e-30)),
        "hydrophone_voltage_rms_v": float(np.sqrt(np.mean(voltage**2))),
        "adc_calibrated_pressure_rms_upa": float(np.sqrt(np.mean(adc_calibrated_pressure**2))) * 1e6,
        "adc_peak_code": int(np.max(np.abs(adc_codes))) if len(adc_codes) else 0,
        "adc_clipping_percent": clipping_percent,
        "sample_count": len(source),
    }
    response: dict[str, Any] = {
        "type": "result",
        "request_id": config["request_id"],
        "job_id": propagation["job_id"],
        "engine": propagation["engine"],
        "config": {
            **{
                key: value for key, value in config.items()
                if key not in {"random_seed", "receiver_noise_spectrum", "source_route"}
            },
            "source_route": [{
                key: value for key, value in point.items() if key != "noise_spectrum"
            } for point in route],
        },
        "noise": {
            "model": config["noise_model"],
            "source_monitor_contains_environment_noise": bool(config["noise_enabled"]),
            "wenz": config.get("wenz_context"),
            "receiver": {
                "lon": config["receiver_lon"],
                "lat": config["receiver_lat"],
                "spectrum_model": (config.get("receiver_noise_spectrum") or {}).get("model"),
                "query_time": (config.get("receiver_noise_spectrum") or {}).get("query_time"),
                "forecast_frame_index": (config.get("receiver_noise_spectrum") or {}).get("forecast_frame_index"),
                "forecast_provider": (config.get("receiver_noise_spectrum") or {}).get("forecast_provider"),
                "drivers": (config.get("receiver_noise_spectrum") or {}).get("drivers"),
            },
            "source_route_positions": len(route),
        },
        "hydrophone_response": {
            key: value for key, value in HYDROPHONE_PRESETS[str(config["hydrophone"])].items()
        },
        "propagation": {
            "distance_m": propagation.get("distance_m"),
            "bearing_deg": propagation.get("bearing_deg"),
            "frequency_hz": propagation.get("frequency_hz"),
            "path_count": len(taps),
            "first_arrival_s": propagation.get("first_arrival_s"),
            "impulse_response": taps,
            "moving_source": len(route_results) > 1,
            "route_distance_m": float(distances[-1]) if len(distances) else 0.0,
            "route_points": route_results,
        },
        "metrics": metrics,
        "waveforms": {
            "source_pressure_pa": _decimate_waveform(time_s, source_monitor),
            "received_pressure_pa": _decimate_waveform(time_s, received),
            "hydrophone_voltage_v": _decimate_waveform(time_s, voltage),
            "adc_normalized": _decimate_waveform(time_s, adc_normalized),
            "adc_calibrated_pressure_pa": _decimate_waveform(time_s, adc_calibrated_pressure),
        },
        "spectrograms": {
            "source_pressure": _spectrogram_payload(
                source_monitor * 1e6,
                int(config["sample_rate"]),
                "dB re 1 µPa²/Hz",
            ),
            "received_pressure": _spectrogram_payload(
                received * 1e6,
                int(config["sample_rate"]),
                "dB re 1 µPa²/Hz",
            ),
            "adc_normalized": _spectrogram_payload(adc_normalized, int(config["sample_rate"])),
            "adc_calibrated_pressure": _spectrogram_payload(
                adc_calibrated_pressure * 1e6,
                int(config["sample_rate"]),
                "dB re 1 µPa²/Hz",
            ),
        },
    }
    response["files"] = _write_result_files(
        jobs_dir,
        str(response["job_id"]),
        int(config["sample_rate"]),
        adc_codes,
        int(config["adc_bits"]),
        adc_normalized,
        source_monitor,
        response,
    )
    _progress(progress, "complete", 100, "信号发射—传播—接收链路计算完成")
    return response


async def serve_signal_transmission_websocket(
    websocket: Any,
    *,
    jobs_dir: str | Path,
    core: Any,
) -> None:
    """处理一个信号传输 WebSocket 会话。"""
    await websocket.accept()
    await websocket.send_text(json.dumps({
        "type": "config_ack",
        "protocol": "signal-transmission-v1",
        "hydrophones": HYDROPHONE_PRESETS,
        "signal_types": ["CW", "TONE_BURST", "LFM"],
        "propagation_models": ["bellhop", "analytic"],
    }, ensure_ascii=False))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                await websocket.send_text(json.dumps({"type": "error", "message": f"JSON 解析失败：{exc}"}, ensure_ascii=False))
                continue
            if not isinstance(request, Mapping):
                await websocket.send_text(json.dumps({"type": "error", "message": "请求必须是 JSON 对象。"}, ensure_ascii=False))
                continue
            message_type = str(request.get("type", "simulate"))
            if message_type == "ping":
                await websocket.send_text('{"type":"pong"}')
                continue
            if message_type != "simulate":
                await websocket.send_text(json.dumps({"type": "error", "message": f"不支持的消息类型：{message_type}"}, ensure_ascii=False))
                continue

            loop = asyncio.get_running_loop()
            if request.get('channel_mode') == 'waypoint':
                from .services.continuous_transmission import run
                cancelled = threading.Event()
                events = asyncio.Queue(maxsize=2)

                def put_event(event):
                    future = asyncio.run_coroutine_threadsafe(events.put(event), loop)
                    while not cancelled.is_set():
                        try:
                            future.result(timeout=0.2)
                            return
                        except TimeoutError:
                            continue
                    future.cancel()

                def produce():
                    try:
                        def report(phase, percent, message):
                            put_event(dict(type='progress', phase=phase, percent=percent, message=message))
                        for event in run(request, jobs_dir, core, report, cancelled):
                            if cancelled.is_set():
                                break
                            put_event(event)
                    except Exception as exc:
                        put_event(dict(type='error', message=str(exc)))
                    finally:
                        put_event(None)

                producer = asyncio.create_task(asyncio.to_thread(produce))
                disconnect = asyncio.create_task(websocket.receive_text())
                try:
                    while True:
                        next_event = asyncio.create_task(events.get())
                        done, _ = await asyncio.wait([next_event, disconnect], return_when=asyncio.FIRST_COMPLETED)
                        if disconnect in done:
                            next_event.cancel()
                            cancelled.set()
                            break
                        event = next_event.result()
                        if event is None:
                            break
                        await websocket.send_text(json.dumps(event, ensure_ascii=False))
                        if event.get('type') == 'chunk':
                            control = json.loads(await disconnect)
                            if control.get('type') != 'continue':
                                cancelled.set()
                                break
                            disconnect = asyncio.create_task(websocket.receive_text())
                finally:
                    cancelled.set()
                    disconnect.cancel()
                    await producer
                return
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            def publish(phase: str, percent: int, message: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "progress", "phase": phase, "percent": percent, "message": message,
                    "request_id": request.get("request_id"),
                })

            task = asyncio.create_task(asyncio.to_thread(
                simulate_signal_transmission,
                request,
                jobs_dir=jobs_dir,
                core=core,
                progress=publish,
            ))
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await websocket.send_text(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            while not queue.empty():
                await websocket.send_text(json.dumps(queue.get_nowait(), ensure_ascii=False, separators=(",", ":")))
            try:
                result = await task
            except Exception as exc:  # 错误需经 WebSocket 返回，不能让连接静默断开。
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "message": str(exc) or exc.__class__.__name__,
                }, ensure_ascii=False))
                continue
            await websocket.send_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    except (WebSocketDisconnect, BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
