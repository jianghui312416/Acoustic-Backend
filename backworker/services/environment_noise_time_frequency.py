"""环境噪声时频图数据生成。

把前端的时频合成场景（风速控制点、降雨区间、船舶经过区间、冰裂事件）
转换为时间 × 频率的功率谱密度矩阵。请求不带 ``timeFrequency`` 时不会调用
本模块，因此原有接口响应结构保持不变。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_DURATION_S = 60.0
DEFAULT_TIME_STEP_S = 0.5
MAX_DURATION_S = 3600.0
MAX_TIME_POINTS = 4000
MAX_FREQUENCY_POINTS = 256
MAX_EVENTS = 50

WIND_SPEED_MPS = np.asarray(
    [0.0, 1.0, 2.45, 4.4, 6.7, 9.35, 12.3, 15.5, 18.95, 22.6, 26.4, 30.5, 35.0]
)
RAIN_RATE_MM_H = np.asarray([0.0, 0.25, 1.0, 2.5, 7.5, 20.0, 50.0])


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数值。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数值。")
    return result


def _level(value: Any, minimum: int, maximum: int, label: str) -> int:
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是整数等级。") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} 必须位于 {minimum} 到 {maximum} 级。")
    return result


def _range(value: Any, default: tuple[float, float], minimum: float, maximum: float) -> tuple[float, float]:
    """接受标量、[min, max] 或 {min, max}，统一为区间。"""
    if value is None:
        low, high = default
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        low = high = float(value)
    elif isinstance(value, Mapping):
        low = _number(value.get("min", default[0]), "区间下限")
        high = _number(value.get("max", low), "区间上限")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if not values:
            low, high = default
        elif len(values) == 1:
            low = high = _number(values[0], "区间值")
        else:
            low = _number(values[0], "区间下限")
            high = _number(values[1], "区间上限")
    else:
        raise ValueError("区间控制量必须是数值、[min,max] 或 {min,max}。")
    if low > high:
        low, high = high, low
    if low < minimum or high > maximum:
        raise ValueError(f"区间控制量必须位于 {minimum:g} 到 {maximum:g}。")
    return float(low), float(high)


def _nearest_level(table: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(table - float(value))))


def _default_wind_points(duration: float, level: int) -> list[dict[str, float]]:
    level = min(12, max(0, int(level)))
    return [
        {"timeS": 0.0, "level": level},
        {"timeS": duration / 3.0, "level": min(12, level + 1)},
        {"timeS": duration * 2.0 / 3.0, "level": min(12, level + 2)},
        {"timeS": duration, "level": level},
    ]


def _default_rain_events(duration: float) -> list[dict[str, float]]:
    scale = duration / 60.0
    return [
        {"startS": 13.5 * scale, "endS": 20.0 * scale, "level": 3},
        {"startS": 38.5 * scale, "endS": 49.5 * scale, "level": 5},
    ]


def _default_ship_events(duration: float, level: int) -> list[dict[str, float]]:
    scale = duration / 60.0
    return [
        {
            "startS": 24.0 * scale,
            "endS": 36.0 * scale,
            "level": min(5, max(1, int(level))),
            "fundamentalHz": 81.0,
            "harmonicCount": 4,
        }
    ]


def _default_crack_events(duration: float) -> list[dict[str, float]]:
    scale = duration / 60.0
    return [
        {"timeS": time_s * scale, "durationS": 1.5, "level": 3}
        for time_s in (9.0, 22.0, 37.0, 51.0)
    ]


def _parse_wind_points(value: Any, duration: float) -> list[dict[str, float]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("windControlPoints 必须是数组。")
    if len(value) > MAX_EVENTS * 4:
        raise ValueError("风速控制点过多。")
    points = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("风速控制点必须是对象。")
        time_s = _number(item.get("timeS", item.get("time", 0.0)), "风速控制点时刻")
        if not 0.0 <= time_s <= duration:
            raise ValueError("风速控制点时刻超出仿真时长。")
        points.append(
            {
                "timeS": time_s,
                "level": _level(item.get("level", 0), 0, 12, "风速控制点等级"),
            }
        )
    return sorted(points, key=lambda point: point["timeS"])


def _parse_interval_events(
    value: Any,
    duration: float,
    label: str,
    *,
    level_minimum: int,
    level_maximum: int,
    extra=None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}必须是数组。")
    if len(value) > MAX_EVENTS:
        raise ValueError(f"{label}不能超过 {MAX_EVENTS} 个。")
    events = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}必须是对象。")
        start = _number(item.get("startS", 0.0), f"{label}开始时间")
        end = _number(item.get("endS", 0.0), f"{label}结束时间")
        if start < 0.0 or end <= start or end > duration:
            raise ValueError(f"{label}时间区间无效。")
        event = {
            "startS": start,
            "endS": end,
            "level": _level(item.get("level", level_minimum), level_minimum, level_maximum, f"{label}等级"),
        }
        if extra:
            extra(event, item, label)
        events.append(event)
    return sorted(events, key=lambda event: event["startS"])


def _parse_crack_events(value: Any, duration: float) -> list[dict[str, float]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("iceCrackEvents 必须是数组。")
    if len(value) > MAX_EVENTS:
        raise ValueError(f"冰裂事件不能超过 {MAX_EVENTS} 个。")
    events = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("冰裂事件必须是对象。")
        time_s = _number(item.get("timeS", 0.0), "冰裂事件时刻")
        duration_s = _number(item.get("durationS", 1.5), "冰裂事件持续时间")
        if not 0.0 <= time_s <= duration:
            raise ValueError("冰裂事件时刻超出仿真时长。")
        if not 0.0 < duration_s <= duration:
            raise ValueError("冰裂事件持续时间无效。")
        events.append(
            {
                "timeS": time_s,
                "durationS": duration_s,
                "level": _level(item.get("level", 3), 1, 5, "冰裂事件等级"),
            }
        )
    return sorted(events, key=lambda event: event["timeS"])


def parse_options(value: Any, *, allow_ice: bool) -> dict[str, Any] | None:
    """解析前端场景；同时兼容风速/降雨/船舶/冰裂的区间写法。"""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("timeFrequency 必须是对象。")
    allowed = {
        "duration", "durationS", "timeStep", "stepS", "timeAxis", "seed",
        "windJitterPercent", "windControlPoints", "rainEvents", "shipEvents",
        "iceCrackEvents", "windSpeed", "rainRate", "shipping", "iceCrack",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("timeFrequency 不支持字段：" + ", ".join(unknown))

    duration = _number(value.get("durationS", value.get("duration", DEFAULT_DURATION_S)), "仿真时长")
    time_step = _number(value.get("stepS", value.get("timeStep", DEFAULT_TIME_STEP_S)), "时间步长")
    if not (1.0 <= duration <= MAX_DURATION_S):
        raise ValueError(f"仿真时长必须位于 1 到 {MAX_DURATION_S:g} 秒。")
    if not (0.05 <= time_step <= 10.0):
        raise ValueError("时间步长必须位于 0.05 到 10 秒。")
    if time_step > duration:
        raise ValueError("时间步长不能大于仿真时长。")
    if math.floor(duration / time_step) + 1 > MAX_TIME_POINTS:
        raise ValueError(f"时频图时间点数不能超过 {MAX_TIME_POINTS}，请增大时间步长。")

    time_axis = str(value.get("timeAxis", "relative")).strip().lower()
    if time_axis not in {"relative", "absolute"}:
        raise ValueError("timeAxis 仅支持 relative 或 absolute。")
    seed = _level(value.get("seed", 2026), -2**31, 2**31 - 1, "随机种子")
    jitter = _number(value.get("windJitterPercent", 6.0), "风速扰动")
    if not 0.0 <= jitter <= 50.0:
        raise ValueError("风速扰动必须位于 0 到 50%。")

    wind_points = _parse_wind_points(value.get("windControlPoints"), duration)
    if not wind_points:
        if value.get("windSpeed") is not None:
            low, high = _range(value.get("windSpeed"), (3.5, 9.8), 0.0, 35.0)
            low_level = _nearest_level(WIND_SPEED_MPS, low)
            high_level = _nearest_level(WIND_SPEED_MPS, high)
            wind_points = [
                {"timeS": 0.0, "level": low_level},
                {"timeS": duration / 2.0, "level": high_level},
                {"timeS": duration, "level": low_level},
            ]
        else:
            wind_points = _default_wind_points(duration, 4)

    rain_events = _parse_interval_events(
        value.get("rainEvents"), duration, "降雨事件", level_minimum=0, level_maximum=6
    )
    if not rain_events and value.get("rainRate") is not None:
        low, high = _range(value.get("rainRate"), (0.0, 30.0), 0.0, 200.0)
        level = _nearest_level(RAIN_RATE_MM_H, max(low, high))
        if level > 0:
            rain_events = [{"startS": duration * 0.25, "endS": duration * 0.75, "level": level}]
    if value.get("rainEvents") is None and value.get("rainRate") is None:
        rain_events = _default_rain_events(duration)

    def ship_extra(event, item, label):
        event["fundamentalHz"] = _number(item.get("fundamentalHz", 81.0), f"{label}基频")
        if not 1.0 <= event["fundamentalHz"] <= 5000.0:
            raise ValueError(f"{label}基频必须位于 1 到 5000 Hz。")
        event["harmonicCount"] = _level(item.get("harmonicCount", 4), 0, 12, f"{label}谐波数量")

    ship_events = _parse_interval_events(
        value.get("shipEvents"), duration, "船舶事件", level_minimum=1, level_maximum=5, extra=ship_extra
    )
    if not ship_events and value.get("shipping") is not None:
        low, high = _range(value.get("shipping"), (0.0, 1.0), 0.0, 1.0)
        level = min(5, max(1, int(round(1 + 4 * max(low, high)))))
        ship_events = [
            {
                "startS": duration * 0.35,
                "endS": duration * 0.65,
                "level": level,
                "fundamentalHz": 81.0,
                "harmonicCount": 4,
            }
        ]
    if value.get("shipEvents") is None and value.get("shipping") is None:
        ship_events = _default_ship_events(duration, 3)

    crack_events = _parse_crack_events(value.get("iceCrackEvents"), duration) if allow_ice else []
    if allow_ice and not crack_events:
        if value.get("iceCrack") is not None:
            low, high = _range(value.get("iceCrack"), (0.0, 1.0), 0.0, 1.0)
            level = min(5, max(1, int(round(1 + 4 * max(low, high)))))
            crack_events = [
                {"timeS": t * duration / 60.0, "durationS": 1.5, "level": level}
                for t in (9.0, 22.0, 37.0, 51.0)
            ]
        else:
            crack_events = _default_crack_events(duration)

    return {
        "durationS": duration,
        "stepS": time_step,
        "timeAxis": time_axis,
        "seed": int(seed),
        "windJitterPercent": jitter,
        "windControlPoints": wind_points,
        "rainEvents": rain_events,
        "shipEvents": ship_events,
        "iceCrackEvents": crack_events,
    }


def _rain_power(freq_hz: np.ndarray, rain_level: int) -> np.ndarray:
    selected = min(6, max(0, int(rain_level)))
    if selected <= 0:
        return np.zeros_like(freq_hz)
    rate = float(RAIN_RATE_MM_H[selected])
    level_at_5k = 50.0 + 17.0 * math.log10(rate)
    peak_frequency = 13500.0
    peak_level = level_at_5k + 15.0 * math.log2(peak_frequency / 5000.0)
    values = np.where(
        freq_hz < peak_frequency,
        level_at_5k + 15.0 * np.log2(np.maximum(freq_hz, 1.0) / 5000.0),
        peak_level - 7.0 * np.log2(freq_hz / peak_frequency),
    )
    return np.power(10.0, values / 10.0)


def _shipping_power(freq_hz: np.ndarray, event: Mapping[str, Any], envelope: float) -> np.ndarray:
    khz = np.maximum(freq_hz / 1000.0, 0.001)
    shipping_factor = (float(event["level"]) - 1.0) / 4.0
    broadband_db = (
        40.0
        + 20.0 * (shipping_factor - 0.5)
        + 26.0 * np.log10(khz)
        - 60.0 * np.log10(khz + 0.03)
    )
    power = np.power(10.0, broadband_db / 10.0) * envelope
    base_frequency = float(event["fundamentalHz"])
    for order in range(1, int(event["harmonicCount"]) + 1):
        center = base_frequency * order
        width = max(1.2, center * 0.015)
        harmonic_db = broadband_db + max(2.0, 16.0 - 3.0 * order)
        power += (
            np.power(10.0, harmonic_db / 10.0)
            * envelope
            * np.exp(-0.5 * ((freq_hz - center) / width) ** 2)
        )
    return power


def _raised_cosine(time_s: float, start: float, end: float, edge: float) -> float:
    if time_s < start or time_s > end:
        return 0.0
    span = max(1e-9, end - start)
    edge = min(max(edge, 1e-6), span / 2.0)
    if time_s < start + edge:
        return 0.5 - 0.5 * math.cos(math.pi * (time_s - start) / edge)
    if time_s > end - edge:
        return 0.5 - 0.5 * math.cos(math.pi * (end - time_s) / edge)
    return 1.0


def _wind_level_at(time_s: float, points: list[dict[str, float]]) -> float:
    if not points:
        return 4.0
    if time_s <= points[0]["timeS"]:
        return float(points[0]["level"])
    if time_s >= points[-1]["timeS"]:
        return float(points[-1]["level"])
    for index in range(1, len(points)):
        current = points[index]
        if time_s <= current["timeS"]:
            previous = points[index - 1]
            span = max(1e-9, current["timeS"] - previous["timeS"])
            ratio = (time_s - previous["timeS"]) / span
            return float(previous["level"] + ratio * (current["level"] - previous["level"]))
    return float(points[-1]["level"])


def _wind_speed_mps(wind_level: float) -> float:
    lower = int(math.floor(wind_level))
    upper = int(math.ceil(wind_level))
    lower = min(12, max(0, lower))
    upper = min(12, max(0, upper))
    ratio = wind_level - math.floor(wind_level)
    return float(WIND_SPEED_MPS[lower] * (1.0 - ratio) + WIND_SPEED_MPS[upper] * ratio)


def _crack_power(freq_hz: np.ndarray, event: Mapping[str, Any], time_s: float) -> np.ndarray:
    duration = max(0.05, float(event["durationS"]))
    sigma = max(0.05, duration / 2.355)
    envelope = math.exp(-0.5 * ((time_s - float(event["timeS"])) / sigma) ** 2)
    spectrum_db = (
        48.0
        + 5.0 * (float(event["level"]) - 1.0)
        + 10.0 * np.log10((300.0 / np.maximum(freq_hz, 20.0)) ** 0.35)
    )
    return np.power(10.0, spectrum_db / 10.0) * envelope


def _limit_frequency(
    frequency_hz: np.ndarray,
    reference_db: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    valid = np.isfinite(frequency) & (frequency > 0.0)
    frequency = frequency[valid]
    reference = None
    if reference_db is not None:
        raw = np.asarray(reference_db).reshape(-1)
        reference = np.asarray(
            [np.nan if value is None else float(value) for value in raw],
            dtype=np.float64,
        )
        if reference.size == valid.size:
            reference = reference[valid]
    if frequency.size > MAX_FREQUENCY_POINTS:
        indices = np.unique(
            np.linspace(0, frequency.size - 1, MAX_FREQUENCY_POINTS).astype(np.int64)
        )
        frequency = frequency[indices]
        if reference is not None and reference.size >= int(indices.max()) + 1:
            reference = reference[indices]
    if frequency.size < 2:
        raise ValueError("时频图至少需要两个有效频率点。")
    return frequency, reference


def build_time_frequency(
    frequency_hz: Any,
    reference_db: Any = None,
    options: Mapping[str, Any] | None = None,
    *,
    ice: bool = False,
) -> dict[str, Any]:
    """生成时间 × 频率的功率谱密度矩阵（dB re 1 µPa²/Hz）。"""
    if options is None:
        raise ValueError("缺少 timeFrequency 参数。")
    frequency, reference = _limit_frequency(np.asarray(frequency_hz), reference_db)
    duration = float(options["durationS"])
    step = float(options["stepS"])
    time_s = np.arange(0.0, duration + step * 0.5, step, dtype=np.float64)
    if time_s.size > MAX_TIME_POINTS:
        time_s = time_s[:MAX_TIME_POINTS]
        time_s[-1] = duration

    rng = np.random.default_rng(int(options["seed"]))
    jitter_percent = float(options["windJitterPercent"]) / 100.0
    smooth = 0.0
    jitter = np.empty(time_s.size, dtype=np.float64)
    for index in range(time_s.size):
        smooth = 0.82 * smooth + 0.18 * (rng.random() * 2.0 - 1.0)
        jitter[index] = smooth

    wind_points = list(options["windControlPoints"])
    rain_events = list(options["rainEvents"])
    ship_events = list(options["shipEvents"])
    crack_events = list(options["iceCrackEvents"]) if ice else []

    psd_db = np.empty((time_s.size, frequency.size), dtype=np.float64)
    for time_index, time_value in enumerate(time_s):
        wind_level = _wind_level_at(float(time_value), wind_points)
        wind_speed = max(0.0, _wind_speed_mps(wind_level) * (1.0 + jitter_percent * jitter[time_index]))
        khz = np.maximum(frequency / 1000.0, 0.001)
        turbulence = np.power(10.0, (17.0 - 30.0 * np.log10(khz)) / 10.0)
        thermal = np.power(10.0, (-15.0 + 20.0 * np.log10(khz)) / 10.0)
        wind = np.power(
            10.0,
            (50.0 + 7.5 * math.sqrt(wind_speed) + 20.0 * np.log10(khz) - 40.0 * np.log10(khz + 0.4))
            / 10.0,
        )
        power = turbulence + thermal + wind

        for event in rain_events:
            envelope = _raised_cosine(
                float(time_value),
                float(event["startS"]),
                float(event["endS"]),
                max(step, (float(event["endS"]) - float(event["startS"])) * 0.12),
            )
            if envelope > 0.0:
                power = power + _rain_power(frequency, int(event["level"])) * envelope

        for event in ship_events:
            envelope = _raised_cosine(
                float(time_value),
                float(event["startS"]),
                float(event["endS"]),
                max(step, (float(event["endS"]) - float(event["startS"])) * 0.2),
            )
            if envelope > 0.0:
                power = power + _shipping_power(frequency, event, envelope)

        for event in crack_events:
            power = power + _crack_power(frequency, event, float(time_value))

        psd_db[time_index] = 10.0 * np.log10(np.maximum(power, 1e-14))

    # 与原接口谱形保持一致：按频率对齐时间平均谱级，仅叠加时变结构。
    if reference is not None and reference.size == frequency.size:
        finite = np.isfinite(reference)
        if np.any(finite):
            offset = np.zeros_like(frequency)
            offset[finite] = reference[finite] - np.mean(psd_db[:, finite], axis=0)
            psd_db += offset

    return {
        "duration": round(duration, 6),
        "timeStep": round(step, 6),
        "timeAxis": options["timeAxis"],
        "time": np.round(time_s, 6).tolist(),
        "frequency": np.round(frequency, 6).tolist(),
        "psd": np.round(psd_db, 3).tolist(),
        "unit": "dB re 1 uPa^2/Hz",
    }
