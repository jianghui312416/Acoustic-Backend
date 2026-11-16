"""从原始 AIS JSONL 构建论文方法约束的全球船舶噪声源网格。

实现 Johnson 等（2026）的核心声源步骤：排除 SOG < 0.5 kn 的位置，
将移动船舶轨迹及驻留时间分配到源单元，在 63 Hz 和 1000 Hz 使用船型
平均单极子源级，最后在线性功率域进行非相干叠加。全球产品采用稀疏
0.1 度网格以兼容现有服务；它是源网格，不包含传播损失或接收声级。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .global_shipping_noise import (
    PROJECT_ROOT,
    _variable,
    cell_area_m2,
    file_sha256,
    nc_path,
    power_to_db,
)


MODEL_VERSION = "ais_shipping_source_grid_article_v2"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "global_shipping_noise_ais.json"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "HYdata"
    / "ais_20261015_03.jsonl"
    / "ais_20261015_03.jsonl"
)
DYNAMIC_MESSAGE_TYPES = {
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
}
EARTH_RADIUS_M = 6_371_000.0


def resolve_jsonl_input(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"AIS JSONL 输入不存在：{path}")
    same_name = path / path.name
    if same_name.is_file():
        return same_name
    candidates = sorted(path.glob("*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"AIS 目录必须恰好包含一个 JSONL 文件：{path}")
    return candidates[0]


def parse_utc_timestamp(value):
    text = str(value or "").strip()
    if not text:
        raise ValueError("时间为空。")
    if text.endswith(" UTC"):
        text = text[:-4].strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if " +0000" in text:
        text = text.replace(" +0000", "+00:00")
    if "." in text:
        head, tail = text.split(".", 1)
        zone_pos = max(tail.find("+"), tail.find("-"))
        if zone_pos >= 0:
            fraction, zone = tail[:zone_pos], tail[zone_pos:]
        else:
            fraction, zone = tail, ""
        text = f"{head}.{fraction[:6]}{zone}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ais_type_group(type_code):
    if type_code is None or not 0 < int(type_code) <= 99:
        return "unavailable"
    code = int(type_code)
    if 20 <= code <= 29:
        return "wing_in_ground"
    if code == 30:
        return "fishing"
    if 31 <= code <= 35 or 50 <= code <= 59:
        return "service"
    if 36 <= code <= 39:
        return "pleasure_or_sailing"
    if 40 <= code <= 49:
        return "high_speed_craft"
    if 60 <= code <= 69:
        return "passenger"
    if 70 <= code <= 79:
        return "cargo"
    if 80 <= code <= 89:
        return "tanker"
    return "other"


def validate_ais_config(config):
    if config.get("model_id") != MODEL_VERSION:
        raise ValueError(f"model_id 必须为 {MODEL_VERSION}。")
    frequency = np.asarray(config["frequency_hz"], dtype=float)
    if frequency.shape != (2,) or not np.allclose(frequency, [63.0, 1000.0]):
        raise ValueError("论文模式严格使用 63 Hz 和 1000 Hz。")
    step = float(config["grid_step_deg"])
    if not 0 < step <= 1 or not np.isclose(1.0 / step, round(1.0 / step)):
        raise ValueError("grid_step_deg 必须能整除 1 度且位于 (0, 1]。")
    for key in (
        "source_depth_m",
        "minimum_sog_kn",
        "maximum_track_gap_seconds",
        "maximum_implied_speed_kn",
        "segment_step_seconds",
    ):
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为正有限数。")
    classes = config["article_class_msl"]
    for name, values in classes.items():
        mean = np.asarray(values["mean_db_re_1upa_m"], dtype=float)
        sd = np.asarray(values["sd_db"], dtype=float)
        if mean.shape != frequency.shape or sd.shape != frequency.shape:
            raise ValueError(f"{name} 的源级频率维度不匹配。")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)) or np.any(sd < 0):
            raise ValueError(f"{name} 的源级参数无效。")
    for group, item in config["ais_type_group_mapping"].items():
        article_class = item["article_class"]
        if article_class is not None and article_class not in classes:
            raise ValueError(f"AIS 类型组 {group} 映射到不存在的论文船型。")
    configured = datetime.fromisoformat(config["time_utc"].replace("Z", "+00:00"))
    if configured.tzinfo is None:
        raise ValueError("time_utc 必须带 UTC 时区。")
    return frequency


def _static_type(row, message_type, payload):
    if message_type == "ShipStaticData":
        value = payload.get("Type")
    elif message_type == "StaticDataReport":
        report_b = payload.get("ReportB") or {}
        value = report_b.get("ShipType") if report_b.get("Valid") else None
    elif message_type == "ExtendedClassBPositionReport":
        value = payload.get("Type")
    else:
        value = None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 99 else None


def _haversine_nm(lon1, lat1, lon2, lat2):
    delta_lon = ((float(lon2) - float(lon1) + 180.0) % 360.0) - 180.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = phi2 - phi1
    dlambda = math.radians(delta_lon)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    distance_m = 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))
    return distance_m / 1852.0


def _selected_type(counter):
    nonzero = [(count, -code, code) for code, count in counter.items() if 0 < code <= 99]
    return max(nonzero)[2] if nonzero else None


def read_ais_jsonl(path, config):
    """读取 AIS 消息、连接静态船型并将移动轨迹的驻留秒数分配到稀疏网格。"""
    validate_ais_config(config)
    path = resolve_jsonl_input(path)
    step = float(config["grid_step_deg"])
    scale = int(round(1.0 / step))
    minimum_sog = float(config["minimum_sog_kn"])
    maximum_gap = float(config["maximum_track_gap_seconds"])
    maximum_speed = float(config["maximum_implied_speed_kn"])
    segment_step = float(config["segment_step_seconds"])
    mappings = config["ais_type_group_mapping"]

    message_types = Counter()
    static_type_candidates = defaultdict(Counter)
    points = defaultdict(list)
    rows = 0
    json_errors = 0
    invalid_dynamic_coordinates = 0
    invalid_dynamic_sog = 0
    below_speed_reports = 0
    valid_dynamic_reports = 0
    observed_start = None
    observed_end = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            message_type = str(row.get("MessageType") or "<missing>")
            message_types[message_type] += 1
            try:
                received_at = parse_utc_timestamp(row.get("received_at"))
            except (TypeError, ValueError):
                continue
            observed_start = received_at if observed_start is None or received_at < observed_start else observed_start
            observed_end = received_at if observed_end is None or received_at > observed_end else observed_end

            meta = row.get("MetaData") or {}
            mmsi = meta.get("MMSI")
            message = row.get("Message") or {}
            payload = message.get(message_type) or {}
            if mmsi is None:
                mmsi = payload.get("UserID")
            try:
                mmsi = int(mmsi)
            except (TypeError, ValueError):
                continue

            type_code = _static_type(row, message_type, payload)
            if type_code is not None:
                static_type_candidates[mmsi][type_code] += 1

            if message_type not in DYNAMIC_MESSAGE_TYPES:
                continue
            lon = payload.get("Longitude", meta.get("longitude"))
            lat = payload.get("Latitude", meta.get("latitude"))
            sog = payload.get("Sog")
            try:
                lon = float(lon)
                lat = float(lat)
            except (TypeError, ValueError):
                invalid_dynamic_coordinates += 1
                continue
            if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
                invalid_dynamic_coordinates += 1
                continue
            try:
                sog = float(sog)
            except (TypeError, ValueError):
                invalid_dynamic_sog += 1
                continue
            if not math.isfinite(sog) or not 0 <= sog < 102.3:
                invalid_dynamic_sog += 1
                continue
            valid_dynamic_reports += 1
            if sog < minimum_sog:
                below_speed_reports += 1
            points[mmsi].append((received_at.timestamp(), lon, lat, sog))

    if json_errors:
        raise ValueError(f"AIS JSONL 有 {json_errors} 行无法解析；拒绝静默丢弃。")
    if observed_start is None or observed_end is None or observed_end <= observed_start:
        raise ValueError("AIS 文件没有有效的时间范围。")
    configured = datetime.fromisoformat(config["time_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    hour_end = configured.timestamp() + 3600.0
    if not (configured.timestamp() <= observed_start.timestamp() < hour_end and observed_end.timestamp() < hour_end + 1.0):
        raise ValueError(
            f"AIS 时间范围 {observed_start.isoformat()}–{observed_end.isoformat()} "
            f"不属于配置小时 {configured.isoformat()}。"
        )
    averaging_period = (observed_end - observed_start).total_seconds()
    if config.get("averaging_period") != "observed_window":
        averaging_period = float(config["averaging_period_seconds"])
    if averaging_period <= 0:
        raise ValueError("平均时段必须为正数。")

    cells = {}
    used_vessels = set()
    active_vessels = set()
    excluded_vessels = set()
    moving_vessels = set()
    type_groups = Counter()
    mapped_classes = Counter()
    mapping_modes = Counter()
    pair_counts = Counter()

    for mmsi, track in points.items():
        track.sort(key=lambda item: item[0])
        is_moving = any(item[3] >= minimum_sog for item in track)
        if is_moving:
            moving_vessels.add(mmsi)
        type_code = _selected_type(static_type_candidates.get(mmsi, Counter()))
        group = ais_type_group(type_code)
        mapping = mappings[group]
        article_class = mapping["article_class"]
        if is_moving:
            type_groups[group] += 1
            mapping_modes[mapping["mapping"]] += 1

        for first, second in zip(track, track[1:]):
            pair_counts["candidate"] += 1
            t1, lon1, lat1, sog1 = first
            t2, lon2, lat2, sog2 = second
            delta_t = t2 - t1
            if delta_t <= 0:
                pair_counts["rejected_nonpositive_time"] += 1
                continue
            if sog1 < minimum_sog or sog2 < minimum_sog:
                pair_counts["rejected_stationary_endpoint"] += 1
                continue
            if delta_t > maximum_gap:
                pair_counts["rejected_track_gap"] += 1
                continue
            implied_speed = _haversine_nm(lon1, lat1, lon2, lat2) / (delta_t / 3600.0)
            if implied_speed > maximum_speed:
                pair_counts["rejected_implied_speed"] += 1
                continue
            pair_counts["used"] += 1
            used_vessels.add(mmsi)
            if article_class is None:
                excluded_vessels.add(mmsi)
            else:
                active_vessels.add(mmsi)

            delta_lon = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
            delta_lat = lat2 - lat1
            subdivisions = max(
                1,
                int(math.ceil(delta_t / segment_step)),
                int(math.ceil(abs(delta_lon) / (step / 2.0))),
                int(math.ceil(abs(delta_lat) / (step / 2.0))),
            )
            duration = delta_t / subdivisions
            for index in range(subdivisions):
                fraction = (index + 0.5) / subdivisions
                longitude = ((lon1 + delta_lon * fraction + 180.0) % 360.0) - 180.0
                latitude = lat1 + delta_lat * fraction
                lon_index = int(round(longitude * scale))
                if lon_index == 180 * scale:
                    lon_index = -180 * scale
                lat_index = int(round(latitude * scale))
                key = (lat_index, lon_index)
                cell = cells.get(key)
                if cell is None:
                    cell = {
                        "occupancy_seconds": 0.0,
                        "included_seconds": 0.0,
                        "excluded_seconds": 0.0,
                        "subsegment_count": 0,
                        "vessels": set(),
                        "included_vessels": set(),
                        "excluded_vessels": set(),
                        "class_seconds": Counter(),
                    }
                    cells[key] = cell
                cell["occupancy_seconds"] += duration
                cell["subsegment_count"] += 1
                cell["vessels"].add(mmsi)
                if article_class is None:
                    cell["excluded_seconds"] += duration
                    cell["excluded_vessels"].add(mmsi)
                else:
                    cell["included_seconds"] += duration
                    cell["included_vessels"].add(mmsi)
                    cell["class_seconds"][article_class] += duration

    if not cells:
        raise ValueError("AIS 数据经航速和轨迹质量过滤后没有可分配到网格的移动轨迹。")

    classes = config["article_class_msl"]
    class_names = list(classes)
    class_index = {name: index for index, name in enumerate(class_names)}
    class_mean = np.asarray([classes[name]["mean_db_re_1upa_m"] for name in class_names], dtype=float)
    class_sd = np.asarray([classes[name]["sd_db"] for name in class_names], dtype=float)
    keys = sorted(cells)
    count = len(keys)
    lon = np.empty(count)
    lat = np.empty(count)
    area = np.empty(count)
    occupancy = np.empty(count)
    included_seconds = np.empty(count)
    excluded_seconds = np.empty(count)
    segment_count = np.empty(count, dtype=np.int32)
    vessel_count = np.empty(count, dtype=np.int32)
    included_vessel_count = np.empty(count, dtype=np.int32)
    excluded_vessel_count = np.empty(count, dtype=np.int32)
    class_seconds = np.zeros((count, len(class_names)))
    source_power = np.zeros((len(config["frequency_hz"]), count))
    sensitivity_low = np.zeros_like(source_power)
    sensitivity_high = np.zeros_like(source_power)

    for index, key in enumerate(keys):
        latitude = key[0] / scale
        longitude = key[1] / scale
        cell = cells[key]
        lat[index], lon[index] = latitude, longitude
        area[index] = cell_area_m2(latitude, step)
        occupancy[index] = cell["occupancy_seconds"]
        included_seconds[index] = cell["included_seconds"]
        excluded_seconds[index] = cell["excluded_seconds"]
        segment_count[index] = cell["subsegment_count"]
        vessel_count[index] = len(cell["vessels"])
        included_vessel_count[index] = len(cell["included_vessels"])
        excluded_vessel_count[index] = len(cell["excluded_vessels"])
        for name, seconds in cell["class_seconds"].items():
            class_id = class_index[name]
            class_seconds[index, class_id] = seconds
            fraction = seconds / averaging_period
            source_power[:, index] += fraction * 10.0 ** (class_mean[class_id] / 10.0)
            sensitivity_low[:, index] += fraction * 10.0 ** ((class_mean[class_id] - class_sd[class_id]) / 10.0)
            sensitivity_high[:, index] += fraction * 10.0 ** ((class_mean[class_id] + class_sd[class_id]) / 10.0)
            mapped_classes[name] += seconds

    arrays = {
        "lon": lon,
        "lat": lat,
        "cell_area_m2": area,
        "track_subsegment_count": segment_count,
        "moving_vessel_count": vessel_count,
        "included_vessel_count": included_vessel_count,
        "excluded_vessel_count": excluded_vessel_count,
        "occupancy_seconds": occupancy,
        "included_vessel_seconds": included_seconds,
        "excluded_vessel_seconds": excluded_seconds,
        "article_class_seconds": class_seconds,
        "cell_source_power_linear": source_power,
        "cell_source_level_db": power_to_db(source_power),
        "cell_source_level_minus_1sd_db": power_to_db(sensitivity_low),
        "cell_source_level_plus_1sd_db": power_to_db(sensitivity_high),
        "article_class_names": class_names,
        "article_class_mean_db": class_mean,
        "article_class_sd_db": class_sd,
    }
    audit = {
        "input_format": "ais_jsonl",
        "input_rows": rows,
        "json_errors": json_errors,
        "message_types": dict(message_types),
        "observed_start_utc": observed_start.isoformat().replace("+00:00", "Z"),
        "observed_end_utc": observed_end.isoformat().replace("+00:00", "Z"),
        "observed_duration_seconds": (observed_end - observed_start).total_seconds(),
        "averaging_period_seconds": averaging_period,
        "valid_dynamic_reports": valid_dynamic_reports,
        "invalid_dynamic_coordinates": invalid_dynamic_coordinates,
        "invalid_dynamic_sog": invalid_dynamic_sog,
        "below_minimum_sog_reports": below_speed_reports,
        "unique_dynamic_mmsi": len(points),
        "unique_moving_mmsi": len(moving_vessels),
        "unique_vessels_with_static_type": sum(
            _selected_type(static_type_candidates.get(mmsi, Counter())) is not None
            for mmsi in points
        ),
        "unique_vessels_with_usable_segments": len(used_vessels),
        "unique_modeled_vessels": len(active_vessels),
        "unique_excluded_vessels_with_segments": len(excluded_vessels),
        "occupied_track_cells": count,
        "active_acoustic_cells": int(np.count_nonzero(included_vessel_count)),
        "track_pair_counts": dict(pair_counts),
        "moving_vessel_type_groups": dict(type_groups),
        "mapping_modes": dict(mapping_modes),
        "modeled_class_exposure_seconds": {name: float(value) for name, value in mapped_classes.items()},
    }
    return arrays, audit, path


def export_ais_netcdf(path, arrays, audit, config, input_path, checksum):
    nsource = arrays["lon"].size
    with Dataset(nc_path(path), "w") as dataset:
        dataset.Conventions = "CF-1.8"
        dataset.model_version = MODEL_VERSION
        dataset.model_status = "uncalibrated_AIS_track_weighted_source_scenario"
        dataset.created_utc = datetime.now(timezone.utc).isoformat()
        dataset.input_format = "raw AIS JSONL"
        dataset.input_path = str(Path(input_path).resolve())
        dataset.input_sha256 = checksum
        dataset.grid_representation = "sparse 0.1-degree cells traversed by quality-controlled moving AIS tracks"
        dataset.source_normalization = "article_monopole_source_level_proxy_dB_re_1_uPa_m"
        dataset.observation_start_utc = audit["observed_start_utc"]
        dataset.observation_end_utc = audit["observed_end_utc"]
        dataset.averaging_period_seconds = audit["averaging_period_seconds"]
        dataset.minimum_sog_kn = float(config["minimum_sog_kn"])
        dataset.article_reference = json.dumps(config["article_reference"], ensure_ascii=False)
        dataset.ais_type_group_mapping_json = json.dumps(config["ais_type_group_mapping"], ensure_ascii=False)
        dataset.audit_json = json.dumps(audit, ensure_ascii=False)
        dataset.limitations_json = json.dumps([
            "Article class-mean source levels measured in Eclipse Sound are not globally calibrated.",
            "Only AIS passenger, cargo, and tanker type groups have direct article source classes.",
            "Fishing, service, pleasure, high-speed, other, and missing AIS types remain in occupancy but are excluded from acoustic power.",
            "Track duration is interpolated only between consecutive moving reports within the configured time and implied-speed limits.",
            "This file is a source grid only; propagation loss and receiver sound level are not included.",
        ], ensure_ascii=False)
        dataset.createDimension("time", 1)
        dataset.createDimension("frequency", len(config["frequency_hz"]))
        dataset.createDimension("source_cell", nsource)
        dataset.createDimension("article_class", len(arrays["article_class_names"]))

        epoch = datetime.fromisoformat(config["time_utc"].replace("Z", "+00:00")).timestamp()
        time_var = _variable(dataset, "time", ("time",), [epoch], "seconds since 1970-01-01 00:00:00 UTC")
        time_var.calendar = "standard"
        _variable(dataset, "frequency_hz", ("frequency",), config["frequency_hz"], "Hz")
        _variable(dataset, "source_cell_id", ("source_cell",), np.arange(nsource), "1")
        _variable(dataset, "lon", ("source_cell",), arrays["lon"], "degrees_east")
        _variable(dataset, "lat", ("source_cell",), arrays["lat"], "degrees_north")
        _variable(dataset, "cell_area_m2", ("source_cell",), arrays["cell_area_m2"], "m2")
        _variable(dataset, "source_depth_m", ("source_cell",), np.full(nsource, config["source_depth_m"]), "m")
        for name in (
            "track_subsegment_count",
            "moving_vessel_count",
            "included_vessel_count",
            "excluded_vessel_count",
        ):
            _variable(dataset, name, ("source_cell",), arrays[name], "1")
        for name in ("occupancy_seconds", "included_vessel_seconds", "excluded_vessel_seconds"):
            _variable(dataset, name, ("source_cell",), arrays[name], "s")
        _variable(dataset, "article_class_seconds", ("source_cell", "article_class"), arrays["article_class_seconds"], "s")
        _variable(dataset, "article_class_mean_msl_db", ("article_class", "frequency"), arrays["article_class_mean_db"], "dB re 1 uPa m")
        _variable(dataset, "article_class_sd_db", ("article_class", "frequency"), arrays["article_class_sd_db"], "dB")
        _variable(dataset, "cell_source_power_linear", ("frequency", "source_cell"), arrays["cell_source_power_linear"], "uPa2 m2")
        _variable(dataset, "cell_source_level_db", ("frequency", "source_cell"), arrays["cell_source_level_db"], "dB re 1 uPa m")
        _variable(dataset, "cell_source_level_minus_1sd_db", ("frequency", "source_cell"), arrays["cell_source_level_minus_1sd_db"], "dB re 1 uPa m")
        _variable(dataset, "cell_source_level_plus_1sd_db", ("frequency", "source_cell"), arrays["cell_source_level_plus_1sd_db"], "dB re 1 uPa m")
        class_names = dataset.createVariable("article_class_name", str, ("article_class",))
        class_names[:] = np.asarray(arrays["article_class_names"], dtype=object)


def build_summary(arrays, audit, config, input_path, checksum, output_dir):
    active = arrays["included_vessel_count"] > 0
    if not np.any(active):
        raise ValueError("没有具备论文对应船型和有效轨迹时长的活跃声源单元。")
    levels = arrays["cell_source_level_db"][:, active]
    active_indices = np.flatnonzero(active)
    ordered = active_indices[np.argsort(arrays["cell_source_level_db"][0, active])[::-1][:10]]
    return {
        "model_version": MODEL_VERSION,
        "model_status": "uncalibrated_AIS_track_weighted_source_scenario",
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": checksum,
        "output_dir": str(Path(output_dir).resolve()),
        "config": config,
        "audit": audit,
        "active_cell_level_statistics_db": {
            str(int(frequency)): {
                "min": float(np.min(levels[index])),
                "median": float(np.median(levels[index])),
                "max": float(np.max(levels[index])),
            }
            for index, frequency in enumerate(config["frequency_hz"])
        },
        "top_63hz_cells": [
            {
                "lon": float(arrays["lon"][index]),
                "lat": float(arrays["lat"][index]),
                "modeled_vessels": int(arrays["included_vessel_count"][index]),
                "modeled_exposure_seconds": float(arrays["included_vessel_seconds"][index]),
                "level_63hz_db": float(arrays["cell_source_level_db"][0, index]),
                "level_1000hz_db": float(arrays["cell_source_level_db"][1, index]),
            }
            for index in ordered
        ],
        "warnings": [
            "源级采用论文在 Eclipse Sound 得到的船型平均值，尚未完成全球校准。",
            "本文件实际观测仅覆盖约 53 分钟；平均功率以实际观测窗口归一化。",
            "只对 AIS 客船、货船和油轮使用论文直接船型；其他类型保留轨迹占用但不计算声功率。",
            "单条位置、超过最大时间间隔或隐含速度异常的轨迹段不推定驻留时间。",
            "结果是源网格，不包含传播损失、海冰、环境背景或接收声级。",
        ],
    }


def export_plot(path, arrays, config, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = arrays["included_vessel_count"] > 0
    occupied = arrays["moving_vessel_count"] > 0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    for frequency_index, (axis, frequency) in enumerate(zip(axes, config["frequency_hz"])):
        axis.scatter(
            arrays["lon"][occupied], arrays["lat"][occupied], s=1.2,
            c="#C8CED7", linewidths=0, label="moving AIS track; unsupported/unknown types included",
        )
        values = arrays["cell_source_level_db"][frequency_index, active]
        vmin, vmax = np.percentile(values, [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.min(values)), float(np.max(values) + 1.0)
        points = axis.scatter(
            arrays["lon"][active], arrays["lat"][active], c=values,
            s=4.0, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0,
        )
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xticks(np.arange(-180, 181, 60))
        axis.set_yticks(np.arange(-90, 91, 30))
        axis.grid(True, color="#D9D9D9", linewidth=0.4)
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
        axis.set_title(f"{frequency:g} Hz time-averaged cell source level")
        axis.legend(loc="lower left", fontsize=7, frameon=True)
        colorbar = fig.colorbar(points, ax=axis, pad=0.01)
        colorbar.set_label("dB re 1 uPa m")
    start = audit["observed_start_utc"].replace("T", " ").replace("Z", "")[:19]
    end = audit["observed_end_utc"].replace("T", " ").replace("Z", "")[:19]
    fig.suptitle(
        "Global AIS-derived distributed shipping-noise source grid\n"
        f"{start} to {end} UTC; SOG >= {config['minimum_sog_kn']:g} kn; 0.1-degree sparse grid",
        fontsize=13,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(path, summary):
    audit = summary["audit"]
    stats = summary["active_cell_level_statistics_db"]
    mappings = summary["config"]["ais_type_group_mapping"]
    mapping_rows = [
        f"| {group} | {item['article_class'] or 'excluded'} | {item['mapping']} |"
        for group, item in mappings.items()
    ]
    text = f"""# 全球 AIS 船舶分布式噪声源网格

输入时间：{audit['observed_start_utc']} 至 {audit['observed_end_utc']}。源网格为稀疏 0.1° 全球网格。

## 结果

- 输入报文：{audit['input_rows']:,}；有效动态报文：{audit['valid_dynamic_reports']:,}。
- `SOG < {summary['config']['minimum_sog_kn']:g} kn` 的动态报文：{audit['below_minimum_sog_reports']:,}。
- 动态 MMSI：{audit['unique_dynamic_mmsi']:,}；移动 MMSI：{audit['unique_moving_mmsi']:,}。
- 有可用轨迹段的船舶：{audit['unique_vessels_with_usable_segments']:,}；进入声功率的船舶：{audit['unique_modeled_vessels']:,}。
- 移动轨迹占用单元：{audit['occupied_track_cells']:,}；活跃声源单元：{audit['active_acoustic_cells']:,}。
- 63 Hz 活跃单元：{stats['63']['min']:.1f}–{stats['63']['max']:.1f} dB，中位数 {stats['63']['median']:.1f} dB re 1 μPa·m。
- 1000 Hz 活跃单元：{stats['1000']['min']:.1f}–{stats['1000']['max']:.1f} dB，中位数 {stats['1000']['median']:.1f} dB re 1 μPa·m。

## 方法

按照 Johnson 等（2026）的核心步骤，排除 SOG 小于 0.5 kn 的位置，在连续移动报文之间插值轨迹并计算网格驻留秒数。5 m源深、63 Hz和1000 Hz船型平均单极子源级保持不变。源单元在观测窗口内的时间平均功率为：

`P_cell(f) = Σ_class [T_class / T_window] × 10^(MSL_class(f)/10)`

所有船舶在功率域非相干叠加。全球产品使用0.1°网格以保持现有服务兼容；论文区域模型使用的是500 m网格。

## AIS船型映射

| AIS类型组 | 论文船型 | 状态 |
|---|---|---|
{chr(10).join(mapping_rows)}

## 限制

{chr(10).join('- ' + warning for warning in summary['warnings'])}
"""
    Path(path).write_text(text, encoding="utf-8")


def build_shipping_source_product(output_path, config, input_path):
    """生成供分布式噪声服务直接加载的 AIS 船舶源 NetCDF。"""
    output_path = Path(output_path)
    arrays, audit, actual_input = read_ais_jsonl(input_path, config)
    checksum = file_sha256(actual_input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        export_ais_netcdf(temporary, arrays, audit, config, actual_input, checksum)
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "output_file": str(output_path.resolve()),
        "time_utc": str(config["time_utc"]),
        "input_file": str(actual_input.resolve()),
        "input_sha256": checksum,
        "active_acoustic_cells": audit["active_acoustic_cells"],
    }


def run(input_path, config_path, output_dir):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖：{output}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    arrays, audit, actual_input = read_ais_jsonl(input_path, config)
    checksum = file_sha256(actual_input)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(arrays, audit, config, actual_input, checksum, output)
    export_ais_netcdf(output / "shipping_sources.nc", arrays, audit, config, actual_input, checksum)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    export_plot(output / "source_grid_overview.png", arrays, config, audit)
    write_report(output / "model_report.md", summary)
    print(json.dumps({
        "output_dir": str(output.resolve()),
        "input_rows": audit["input_rows"],
        "unique_moving_mmsi": audit["unique_moving_mmsi"],
        "unique_modeled_vessels": audit["unique_modeled_vessels"],
        "occupied_track_cells": audit["occupied_track_cells"],
        "active_acoustic_cells": audit["active_acoustic_cells"],
    }, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="从原始 AIS JSONL 生成全球船舶分布式噪声源网格。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "shipping_sources_ais_20261015_03_v2"),
    )
    args = parser.parse_args()
    run(args.input, args.config, args.output_dir)


if __name__ == "__main__":
    main()
