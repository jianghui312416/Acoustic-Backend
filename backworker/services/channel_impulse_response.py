"""Bellhop 原生信道冲激响应服务。

按指定的发射/接收经纬度、双方深度以及俯角/仰角范围运行 Bellhop arrivals，
输出点对点水声通信信道的原生复数冲激响应。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..array_fusion_service import (
    BellhopFieldSnapshot,
    _friendly_arrival_solver_error,
    _parse_bellhop_ascii_arrivals,
)
from ..bellhop_model.config import load_config as load_bellhop_config
from ..bellhop_model.runner import _run_solver


NM_TO_KM = 1.852
DEFAULT_ARRIVAL_BEAMS = 2001


def _finite_or_none(value: Any):
    number = float(value)
    return round(number, 9) if math.isfinite(number) else None


def _wrap_bearing(degrees: float) -> float:
    return float(degrees) % 360.0


def _distance_and_bearing_m(
    source_lon: float,
    source_lat: float,
    receiver_lon: float,
    receiver_lat: float,
) -> tuple[float, float]:
    latitude_1 = math.radians(source_lat)
    latitude_2 = math.radians(receiver_lat)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = math.radians(receiver_lon - source_lon)
    haversine = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2.0) ** 2
    )
    central_angle = 2.0 * math.atan2(
        math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine))
    )
    distance_m = 6_371_008.8 * central_angle
    east = math.sin(delta_longitude) * math.cos(latitude_2)
    north = (
        math.cos(latitude_1) * math.sin(latitude_2)
        - math.sin(latitude_1) * math.cos(latitude_2) * math.cos(delta_longitude)
    )
    return distance_m, _wrap_bearing(math.degrees(math.atan2(east, north)))


def _angle_between_deg(first_deg: float, second_deg: float) -> float:
    return abs((float(first_deg) - float(second_deg) + 180.0) % 360.0 - 180.0)


def _requested_frequency(payload: Mapping[str, Any], core: Any) -> float:
    if any(name in payload for name in ("frequencyHz", "frequency_hz", "frequency", "f0", "carrierHz")):
        return core._api_float(
            payload,
            ("frequencyHz", "frequency_hz", "frequency", "f0", "carrierHz"),
            100.0,
            "载波频率",
        )
    minimum = core._api_float(payload, ("minFreq",), 100.0, "最小频率")
    maximum = core._api_float(payload, ("maxFreq",), minimum, "最大频率")
    if minimum > maximum:
        raise core.PortalRequestError("最小频率不能大于最大频率。")
    return 0.5 * (minimum + maximum)


def _coordinate_pair(
    payload: Mapping[str, Any],
    core: Any,
    lon_names: Sequence[str],
    lat_names: Sequence[str],
    label: str,
) -> tuple[float, float]:
    for names, suffix in ((lon_names, "经度"), (lat_names, "纬度")):
        if not any(name in payload and payload[name] not in (None, "") for name in names):
            raise core.PortalRequestError(f"缺少必填字段：{label}{suffix}。")
    lon = core._api_float(payload, lon_names, 0.0, f"{label}经度")
    lat = core._api_float(payload, lat_names, 0.0, f"{label}纬度")
    if not -180.0 <= lon <= 180.0:
        raise core.PortalRequestError(f"{label}经度必须位于 -180 到 180°。")
    if not -90.0 <= lat <= 90.0:
        raise core.PortalRequestError(f"{label}纬度必须位于 -90 到 90°。")
    return lon, lat


def _require_field(
    payload: Mapping[str, Any],
    names: Sequence[str],
    label: str,
    core: Any,
) -> None:
    if not any(name in payload and payload[name] not in (None, "") for name in names):
        raise core.PortalRequestError(f"缺少必填字段：{label}。")


def _format_bellhop_values(values: Sequence[float], decimals: int, per_line: int = 8) -> list[str]:
    formatted: list[str] = []
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        suffix = " /" if start + per_line >= len(values) else ""
        formatted.append(" ".join(f"{float(value):.{decimals}f}" for value in chunk) + suffix)
    return formatted


def _write_arrival_environment(
    template_path: Path,
    target_path: Path,
    receiver_depth_m: float,
    range_km: float,
    beam_count: int,
    alpha_min_deg: float,
    alpha_max_deg: float,
) -> None:
    """在已有 Bellhop TL ENV 上替换为单接收点并切换为 A 模式。"""
    lines = template_path.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 15:
        raise ValueError(f"Bellhop ENV 结构不完整：{template_path}")
    tail = lines[-10:]
    try:
        source_count = int(tail[0].strip())
        float(tail[1].split()[0])
        int(tail[2].strip())
        int(tail[4].strip())
        int(tail[7].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"无法识别 Bellhop ENV 接收网格：{template_path}") from exc
    if source_count != 1:
        raise ValueError("信道冲激响应接口要求 Bellhop ENV 只有一个声源深度。")
    if receiver_depth_m < 0.0 or range_km <= 0.0:
        raise ValueError("接收深度必须非负，接收距离必须为正。")
    replacement = [
        "1",
        tail[1],
        "1",
        f"{float(receiver_depth_m):.6f} /",
        "1",
        f"{float(range_km):.9f} /",
        "'AG'",
        str(int(beam_count)),
        f"{float(alpha_min_deg):.6f} {float(alpha_max_deg):.6f} /",
        tail[9],
    ]
    target_path.write_text(
        "\n".join([*lines[:-10], *replacement]) + "\n", encoding="ascii"
    )


def _template_bearing(path: Path) -> float | None:
    first_line = path.read_text(encoding="ascii", errors="replace").splitlines()[0]
    match = re.search(r"bearing\s+([-+0-9.]+)\s+deg", first_line, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _find_template_environment(job_dir: Path, bearing_deg: float) -> Path:
    candidates = sorted(job_dir.rglob("Point2D_TL.env"))
    if not candidates:
        candidates = sorted(job_dir.rglob("Point2D.env"))
    if not candidates:
        raise FileNotFoundError("Bellhop 任务未保留二维 ENV，无法生成原生 arrivals。")
    return min(
        candidates,
        key=lambda path: _angle_between_deg(
            _template_bearing(path) if _template_bearing(path) is not None else bearing_deg,
            bearing_deg,
        ),
    )


def _generate_arrivals(
    job_dir: Path,
    template_path: Path,
    receiver_depth_m: float,
    range_km: float,
    beam_count: int,
    alpha_min_deg: float,
    alpha_max_deg: float,
) -> dict[str, Any]:
    """在任务目录内以单接收点运行 Bellhop A 模式并缓存结果。"""
    bellhop_config = load_bellhop_config()
    executable = str(bellhop_config["BELLHOP_CXX"])
    timeout_sec = float(bellhop_config.get("MAX_WAIT_TIME", 120.0))
    signature_data = {
        "job_id": job_dir.name,
        "template": str(template_path.relative_to(job_dir)),
        "depth_m": round(float(receiver_depth_m), 6),
        "range_km": round(float(range_km), 9),
        "beam_count": int(beam_count),
        "alpha_deg": [round(float(alpha_min_deg), 6), round(float(alpha_max_deg), 6)],
        "solver": executable,
        "solver_mtime_ns": Path(executable).stat().st_mtime_ns if Path(executable).is_file() else None,
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    work_dir = job_dir / "communication_channel" / f"case_{signature}"
    work_dir.mkdir(parents=True, exist_ok=True)
    root_name = "ChannelArrivals"
    arrival_path = work_dir / f"{root_name}.arr"
    failure_path = work_dir / "generation_error.txt"
    if failure_path.is_file() and not arrival_path.is_file():
        raise RuntimeError(
            _friendly_arrival_solver_error(failure_path.read_text(encoding="utf-8", errors="replace"))
        )
    if not arrival_path.is_file():
        _write_arrival_environment(
            template_path,
            work_dir / f"{root_name}.env",
            receiver_depth_m,
            range_km,
            beam_count,
            alpha_min_deg,
            alpha_max_deg,
        )
        template_root = template_path.with_suffix("")
        for extension in (".ssp", ".bty", ".ati", ".trc"):
            source_path = template_root.with_suffix(extension)
            if source_path.is_file():
                shutil.copy2(source_path, work_dir / f"{root_name}{extension}")
        try:
            _run_solver(executable, root_name, work_dir, timeout_sec, True)
        except (OSError, RuntimeError) as exc:
            message = _friendly_arrival_solver_error(exc)
            failure_path.write_text(message, encoding="utf-8")
            raise RuntimeError(message) from exc
        if not arrival_path.is_file():
            message = "Bellhop arrivals 计算结束但未生成 .arr 文件"
            failure_path.write_text(message, encoding="utf-8")
            raise RuntimeError(message)
    parsed = _parse_bellhop_ascii_arrivals(arrival_path)
    parsed['raw_arrival_file'] = str(arrival_path.relative_to(job_dir))
    parsed['raw_arrival_sha256'] = hashlib.sha256(arrival_path.read_bytes()).hexdigest()
    return parsed


def _records_at_point(
    records: Sequence[Mapping[str, Any]],
    depth_m: float,
    range_km: float,
) -> list[dict[str, Any]]:
    target_m = range_km * 1000.0
    metre_tolerance = max(1.0e-6, 1.0e-6 * abs(target_m))
    kilometre_tolerance = max(1.0e-9, 1.0e-6 * abs(range_km))
    selected = []
    for record in records:
        raw_range = float(record["raw_range"])
        range_matches = (
            abs(raw_range - target_m) <= metre_tolerance
            or abs(raw_range - range_km) <= kilometre_tolerance
        )
        if (
            abs(float(record["receiver_depth_m"]) - depth_m) <= 1.0e-6
            and range_matches
        ):
            selected.append(dict(record))
    return selected


def _build_impulse_response(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    taps = []
    for record in records:
        amplitude = float(record["amplitude"])
        phase_deg = float(record["phase_deg"])
        gain = amplitude * complex(
            math.cos(math.radians(phase_deg)), math.sin(math.radians(phase_deg))
        )
        taps.append({
            "arrival_index": record.get("arrival_index"),
            "delay_s": round(float(record["delay_s"]), 9),
            "delay_imag_s": _finite_or_none(record.get("delay_imag_s", 0.0)),
            "amplitude": round(amplitude, 12),
            "phase_deg": round(phase_deg, 6),
            "gain_real": round(gain.real, 12),
            "gain_imag": round(gain.imag, 12),
            "source_angle_deg": _finite_or_none(record.get("source_angle_deg", math.nan)),
            "arrival_angle_deg": _finite_or_none(record.get("arrival_angle_deg", math.nan)),
            "top_bounces": record.get("top_bounces"),
            "bottom_bounces": record.get("bottom_bounces"),
        })
    return sorted(taps, key=lambda item: item["delay_s"])


def _run_portal_job_for_point(
    payload: Mapping[str, Any],
    core: Any,
    source_lon: float,
    source_lat: float,
    source_depth_m: float,
    frequency_hz: float,
    distance_km: float,
    bearing_deg: float,
    receiver_depth_m: float,
) -> tuple[str, Path, BellhopFieldSnapshot]:
    portal_request = {
        "coordX": source_lon,
        "coordY": source_lat,
        "srcDepth": source_depth_m,
        "range": (distance_km + 1.0) / NM_TO_KM,
        "azimuth": bearing_deg,
        "minFreq": int(round(frequency_hz)),
        "maxFreq": int(round(frequency_hz)),
        "grazingAngle": 0.0,
        "beamWidth": 180.0,
        "receiverDepthMax": receiver_depth_m,
        "dataSource": core._pick(payload, ("dataSource",), "EnvReanalysis"),
        "dataType": core._pick(payload, ("dataType",), "DT_ENV_SOUND"),
        "time": core._pick(payload, ("time",), core._portal_default("time", "")),
        "timeRes": core._pick(payload, ("timeRes",), 0),
        "coherent": False,
        "isWaterChanged": True,
    }
    params, _row, _col = core._portal_2d_tl_params(portal_request)
    params.update({"n_rd": 2, "n_range": 2, "tl_n_rd": 2, "tl_n_range": 2})
    job_id, job_dir, _ = core._run_portal_job(params)
    field = BellhopFieldSnapshot.load(core.JOBS_DIR, job_id)
    return job_id, Path(job_dir), field


def build_channel_impulse_response(
    payload: Mapping[str, Any],
    core: Any,
    plugin_name: str = "AnalyseChannelImpulseResponse",
) -> tuple[dict[str, Any], str]:
    """运行 Bellhop arrivals 并返回指定收发点的原生信道冲激响应。"""
    if not isinstance(payload, Mapping):
        raise core.PortalRequestError("请求体必须是 JSON 对象。")

    source_lon, source_lat = _coordinate_pair(
        payload, core,
        ("sourceLon", "coordX", "source_lon"),
        ("sourceLat", "coordY", "source_lat"),
        "发射点",
    )
    receiver_lon, receiver_lat = _coordinate_pair(
        payload, core,
        ("receiverLon", "receiver_lon"),
        ("receiverLat", "receiver_lat"),
        "接收点",
    )
    _require_field(payload, ("sourceDepth", "srcDepth", "source_depth_m"), "发射深度", core)
    _require_field(payload, ("receiverDepth", "receiver_depth_m"), "接收深度", core)
    _require_field(
        payload,
        ("depressionAngleDeg", "depression_angle_deg", "downAngleDeg", "grazingDownDeg"),
        "俯角", core,
    )
    _require_field(
        payload,
        ("elevationAngleDeg", "elevation_angle_deg", "upAngleDeg", "grazingUpDeg"),
        "仰角", core,
    )
    source_depth = core._api_float(
        payload, ("sourceDepth", "srcDepth", "source_depth_m"), 100.0, "发射深度"
    )
    receiver_depth = core._api_float(
        payload, ("receiverDepth", "receiver_depth_m"), 100.0, "接收深度"
    )
    if source_depth <= 0.0 or receiver_depth <= 0.0:
        raise core.PortalRequestError("发射深度和接收深度必须为正数。")

    depression = core._api_float(
        payload,
        ("depressionAngleDeg", "depression_angle_deg", "downAngleDeg", "grazingDownDeg"),
        45.0,
        "俯角",
    )
    elevation = core._api_float(
        payload,
        ("elevationAngleDeg", "elevation_angle_deg", "upAngleDeg", "grazingUpDeg"),
        45.0,
        "仰角",
    )
    depression = abs(depression)
    elevation = abs(elevation)
    if depression > 90.0 or elevation > 90.0:
        raise core.PortalRequestError("俯角和仰角必须位于 0 到 90°。")
    if depression == 0.0 and elevation == 0.0:
        raise core.PortalRequestError("俯角和仰角不能同时为 0。")
    alpha_min_deg = -depression
    alpha_max_deg = elevation

    _require_field(
        payload,
        ("frequencyHz", "frequency_hz", "frequency", "f0", "carrierHz", "minFreq", "maxFreq"),
        "频率", core,
    )
    frequency_hz = _requested_frequency(payload, core)
    beam_count = core._api_int(
        payload, ("arrivalBeamCount", "arrival_beam_count", "nbeams"),
        DEFAULT_ARRIVAL_BEAMS, "arrivals 波束数",
    )
    if beam_count < 1:
        raise core.PortalRequestError("arrivalBeamCount 必须为正整数。")

    distance_m, bearing_deg = _distance_and_bearing_m(
        source_lon, source_lat, receiver_lon, receiver_lat
    )
    if distance_m <= 0.0:
        raise core.PortalRequestError("发射点和接收点不能重合。")
    distance_km = distance_m / 1000.0

    job_id, job_dir, field = _run_portal_job_for_point(
        payload, core, source_lon, source_lat, source_depth,
        frequency_hz, distance_km, bearing_deg, receiver_depth,
    )
    if math.isfinite(field.frequency_hz) and field.frequency_hz > 0.0:
        frequency_hz = float(field.frequency_hz)
    _tl_db, bottom_depth = field.sample(distance_km, receiver_depth)
    if not math.isfinite(bottom_depth):
        raise core.PortalRequestError("接收点超出 Bellhop 声场范围。")
    if receiver_depth >= bottom_depth - 0.5:
        raise core.PortalRequestError(
            f"接收深度 {receiver_depth:g} m 已到达或超过海底 {bottom_depth:g} m。"
        )

    template_path = _find_template_environment(job_dir, bearing_deg)
    arrivals = _generate_arrivals(
        job_dir, template_path, receiver_depth, distance_km, beam_count,
        alpha_min_deg, alpha_max_deg,
    )
    records = _records_at_point(arrivals["records"], receiver_depth, distance_km)
    if not records and not payload.get('allowEmptyArrivals', False):
        raise core.PortalRequestError(
            "该收发点在所选俯角/仰角范围内没有 Bellhop 声到达；"
            "请检查角度范围、深度或增大 arrivalBeamCount。"
        )
    impulse_response = _build_impulse_response(records)
    return {
        "job_id": job_id,
        "plugin": plugin_name,
        "engine": "Bellhop arrivals",
        "source": {
            "lon": round(source_lon, 8),
            "lat": round(source_lat, 8),
            "depth_m": round(source_depth, 6),
        },
        "receiver": {
            "lon": round(receiver_lon, 8),
            "lat": round(receiver_lat, 8),
            "depth_m": round(receiver_depth, 6),
        },
        "distance_m": round(distance_m, 6),
        "bearing_deg": round(bearing_deg, 6),
        "frequency_hz": round(frequency_hz, 6),
        "angle_range_deg": {
            "depression": round(depression, 6),
            "elevation": round(elevation, 6),
            "min": round(alpha_min_deg, 6),
            "max": round(alpha_max_deg, 6),
        },
        "path_count": len(impulse_response),
        "first_arrival_s": (
            impulse_response[0]["delay_s"] if impulse_response else None
        ),
        "impulse_response": impulse_response,
        "raw_arrivals": records,
        "raw_arrival_file": arrivals.get("raw_arrival_file"),
        "raw_arrival_sha256": arrivals.get("raw_arrival_sha256"),
    }, job_id
