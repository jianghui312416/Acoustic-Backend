"""根据 GFW 每小时船舶存在数据构建稀疏的全球船舶噪声源网格。

首版遵循 Johnson 等（2026）的声源网格概念：移动船舶网格单元、63/1000 Hz
单极子声源级、5 m 声源深度、时长加权以及线性功率非相干叠加。提供的 GFW
文件不含航速、船长或轨迹信息，因此使用文中各船型均值作为明确记录的替代值，
且无法过滤静止船舶。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
from netCDF4 import Dataset

EARTH_RADIUS_M = 6_371_000.0
MODEL_VERSION = "gfw_shipping_source_grid_article_proxy_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "global_shipping_noise.json"


def nc_path(path):
    """兼容 Windows NetCDF C 库不能直接打开中文绝对路径的情况。"""
    try:
        return os.path.relpath(Path(path).resolve())
    except ValueError:
        return str(path)


def power_to_db(power):
    a = np.asarray(power, dtype=float)
    if np.any(a < 0):
        raise ValueError("线性声功率不能为负数。")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(a)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_presence_text(path):
    """根据文件签名打开普通 CSV 或 gzip 压缩的 CSV。"""
    path = Path(path)
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def cell_area_m2(latitude_deg, grid_step_deg):
    south = max(-90.0, latitude_deg - grid_step_deg / 2.0)
    north = min(90.0, latitude_deg + grid_step_deg / 2.0)
    width = math.radians(grid_step_deg)
    return EARTH_RADIUS_M**2 * width * (
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )


def validate_config(config):
    if config.get("model_id") != MODEL_VERSION:
        raise ValueError(f"model_id 必须为 {MODEL_VERSION}。")
    frequency = np.asarray(config["frequency_hz"], dtype=float)
    if frequency.shape != (2,) or not np.allclose(frequency, [63.0, 1000.0]):
        raise ValueError("首版严格使用文章的 63 Hz 和 1000 Hz。")
    step = float(config["grid_step_deg"])
    if not np.isclose(step, 0.1):
        raise ValueError("输入为 GFW LOW 0.1° 网格，grid_step_deg 必须为 0.1。")
    if float(config["averaging_period_seconds"]) <= 0:
        raise ValueError("averaging_period_seconds 必须为正数。")
    if float(config["source_depth_m"]) <= 0:
        raise ValueError("source_depth_m 必须为正数。")
    classes = config["article_class_msl"]
    for name, values in classes.items():
        mean = np.asarray(values["mean_db_re_1upa_m"], dtype=float)
        sd = np.asarray(values["sd_db"], dtype=float)
        if mean.shape != frequency.shape or sd.shape != frequency.shape:
            raise ValueError(f"{name} 的频率参数长度不匹配。")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sd)) or np.any(sd < 0):
            raise ValueError(f"{name} 的 MSL 参数无效。")
    mapping = config["gfw_vessel_type_mapping"]
    for vessel_type, item in mapping.items():
        cls = item["article_class"]
        if cls is not None and cls not in classes:
            raise ValueError(f"{vessel_type} 映射到不存在的文章船型 {cls}。")
    return frequency


def read_presence(path, config):
    """返回稀疏网格单元和审计统计信息，不将全部行载入内存。"""
    step = float(config["grid_step_deg"])
    scale = int(round(1.0 / step))
    mapping = config["gfw_vessel_type_mapping"]
    classes = config["article_class_msl"]
    frequency = np.asarray(config["frequency_hz"], dtype=float)
    period = float(config["averaging_period_seconds"])
    expected_date = config["time_utc"].replace("T", " ")[:16]

    cells = {}
    vessel_types = Counter()
    mapped_classes = Counter()
    mapping_modes = Counter()
    seen_vessels = set()
    duplicate_vessel_ids = 0
    dates = Counter()
    rows = 0
    invalid_rows = 0

    with open_presence_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"date", "lon", "lat", "hours", "vesselId", "vesselType"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"输入缺少字段: {sorted(missing)}")
        for row in reader:
            rows += 1
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
                hours = float(row["hours"])
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if not (-180 <= lon < 180 and -90 <= lat <= 90 and math.isfinite(hours) and hours >= 0):
                invalid_rows += 1
                continue
            stamp = str(row["date"]).replace("T", " ")[:16]
            dates[stamp] += 1
            vessel_id = row["vesselId"]
            if vessel_id in seen_vessels:
                duplicate_vessel_ids += 1
            elif vessel_id:
                seen_vessels.add(vessel_id)

            vessel_type = (row["vesselType"] or "").strip().upper()
            vessel_types[vessel_type] += 1
            item = mapping.get(vessel_type)
            if item is None:
                item = {"article_class": None, "mapping": "excluded_unmapped_type"}
            mapping_modes[item["mapping"]] += 1

            # GFW 将经纬度定义为网格单元中心。将 float32 产生的误差值
            # （例如 119.4000015）舍入到文档规定的 0.1° 网格中心。
            lon_index = int(round(lon * scale))
            lat_index = int(round(lat * scale))
            key = (lat_index, lon_index)
            cell = cells.get(key)
            if cell is None:
                cell = {
                    "record_count": 0,
                    "hours": 0.0,
                    "included_count": 0,
                    "included_hours": 0.0,
                    "proxy_count": 0,
                    "excluded_count": 0,
                    "type_counts": Counter(),
                    "class_hours": Counter(),
                }
                cells[key] = cell
            cell["record_count"] += 1
            cell["hours"] += hours
            cell["type_counts"][vessel_type] += 1
            cls = item["article_class"]
            if cls is None:
                cell["excluded_count"] += 1
                continue
            cell["included_count"] += 1
            cell["included_hours"] += hours
            if item["mapping"] != "direct":
                cell["proxy_count"] += 1
            cell["class_hours"][cls] += hours
            mapped_classes[cls] += 1

    if invalid_rows:
        raise ValueError(f"存在 {invalid_rows} 条无效坐标/小时记录；首版拒绝静默丢弃。")
    if rows == 0 or not cells:
        raise ValueError("输入没有可建模记录。")
    if list(dates) != [expected_date]:
        raise ValueError(f"时间与配置不一致: 配置 {expected_date}, 实际 {dict(dates)}")
    if duplicate_vessel_ids:
        raise ValueError(f"同一小时发现 {duplicate_vessel_ids} 个重复 vesselId；需先确定去重规则。")

    class_names = list(classes)
    class_mean = np.asarray([classes[n]["mean_db_re_1upa_m"] for n in class_names], dtype=float)
    class_sd = np.asarray([classes[n]["sd_db"] for n in class_names], dtype=float)
    class_index = {name: i for i, name in enumerate(class_names)}
    vessel_type_names = sorted(vessel_types, key=lambda x: (x == "", x))

    keys = sorted(cells)
    count = len(keys)
    lon = np.empty(count)
    lat = np.empty(count)
    area = np.empty(count)
    records = np.empty(count, dtype=np.int32)
    vessel_hours = np.empty(count)
    included = np.empty(count, dtype=np.int32)
    included_hours = np.empty(count)
    proxy_count = np.empty(count, dtype=np.int32)
    excluded = np.empty(count, dtype=np.int32)
    type_count = np.zeros((count, len(vessel_type_names)), dtype=np.int32)
    class_hours = np.zeros((count, len(class_names)))
    source_power = np.zeros((len(frequency), count))
    sensitivity_low = np.zeros_like(source_power)
    sensitivity_high = np.zeros_like(source_power)

    type_index = {name: i for i, name in enumerate(vessel_type_names)}
    for i, key in enumerate(keys):
        latitude = key[0] / scale
        longitude = key[1] / scale
        cell = cells[key]
        lat[i], lon[i] = latitude, longitude
        area[i] = cell_area_m2(latitude, step)
        records[i] = cell["record_count"]
        vessel_hours[i] = cell["hours"]
        included[i] = cell["included_count"]
        included_hours[i] = cell["included_hours"]
        proxy_count[i] = cell["proxy_count"]
        excluded[i] = cell["excluded_count"]
        for name, n in cell["type_counts"].items():
            type_count[i, type_index[name]] = n
        for name, h in cell["class_hours"].items():
            j = class_index[name]
            class_hours[i, j] = h
            fraction = h * 3600.0 / period
            source_power[:, i] += fraction * 10.0 ** (class_mean[j] / 10.0)
            sensitivity_low[:, i] += fraction * 10.0 ** ((class_mean[j] - class_sd[j]) / 10.0)
            sensitivity_high[:, i] += fraction * 10.0 ** ((class_mean[j] + class_sd[j]) / 10.0)

    arrays = {
        "lon": lon,
        "lat": lat,
        "cell_area_m2": area,
        "presence_record_count": records,
        "vessel_hours": vessel_hours,
        "included_vessel_count": included,
        "included_vessel_hours": included_hours,
        "proxy_vessel_count": proxy_count,
        "excluded_record_count": excluded,
        "vessel_type_count": type_count,
        "article_class_hours": class_hours,
        "cell_source_power_linear": source_power,
        "cell_source_level_db": power_to_db(source_power),
        "cell_source_level_minus_1sd_db": power_to_db(sensitivity_low),
        "cell_source_level_plus_1sd_db": power_to_db(sensitivity_high),
        "vessel_type_names": vessel_type_names,
        "article_class_names": class_names,
        "article_class_mean_db": class_mean,
        "article_class_sd_db": class_sd,
    }
    audit = {
        "input_rows": rows,
        "unique_vessel_ids": len(seen_vessels),
        "occupied_presence_cells": count,
        "active_acoustic_cells": int(np.count_nonzero(included)),
        "included_records": int(included.sum()),
        "excluded_records": int(excluded.sum()),
        "proxy_records": int(proxy_count.sum()),
        "direct_records": int(included.sum() - proxy_count.sum()),
        "input_vessel_types": dict(vessel_types),
        "mapped_article_classes": dict(mapped_classes),
        "mapping_modes": dict(mapping_modes),
        "date_counts": dict(dates),
    }
    return arrays, audit


def _variable(ds, name, dimensions, data, units="1", long_name=""):
    a = np.asarray(data)
    dtype = "i4" if a.dtype.kind in "biu" and a.dtype.itemsize <= 4 else "i8" if a.dtype.kind in "iu" else "f8"
    fill = None if dtype.startswith("i") else np.nan
    kwargs = {"zlib": True, "complevel": 4}
    if fill is not None:
        kwargs["fill_value"] = fill
    var = ds.createVariable(name, dtype, dimensions, **kwargs)
    # netCDF4 1.7.x 在准备写入时仍会修改内部 NumPy shape 数组。即使变量形状
    # 与输入形状一致，NumPy 2.5 仍会针对这一实现细节发出警告，因此仅在此处
    # 局部屏蔽该兼容性警告。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        var[:] = a
    var.units = units
    if long_name:
        var.long_name = long_name
    return var


def export_netcdf(path, arrays, audit, config, input_path, checksum):
    nsource = arrays["lon"].size
    with Dataset(nc_path(path), "w") as ds:
        ds.Conventions = "CF-1.8"
        ds.model_version = MODEL_VERSION
        ds.model_status = "uncalibrated_global_engineering_source_scenario"
        ds.created_utc = datetime.now(timezone.utc).isoformat()
        ds.input_path = str(Path(input_path).resolve())
        ds.input_sha256 = checksum
        ds.grid_representation = "sparse list of non-empty GFW LOW 0.1-degree cell centers"
        ds.source_normalization = "article_monopole_source_level_proxy_dB_re_1_uPa_m"
        ds.article_reference = json.dumps(config["article_reference"], ensure_ascii=False)
        ds.vessel_type_mapping_json = json.dumps(config["gfw_vessel_type_mapping"], ensure_ascii=False)
        ds.limitations_json = json.dumps([
            "GFW input has no speed, length, heading, or track; article class means are proxies.",
            "Article source levels are local Eclipse Sound observations and are not globally calibrated.",
            "The article excludes SOG below 0.5 kn; this input has no speed, so stationary vessels cannot be removed.",
            "Seismic-vessel values represent propulsion proxy only; impulsive airgun noise is not modeled.",
            "GEAR and unresolved records are retained in occupancy counts but excluded from acoustic power.",
            "This file is a source grid only; no propagation loss or receiver sound level is included."
        ], ensure_ascii=False)
        ds.audit_json = json.dumps(audit, ensure_ascii=False)
        ds.createDimension("time", 1)
        ds.createDimension("frequency", len(config["frequency_hz"]))
        ds.createDimension("source_cell", nsource)
        ds.createDimension("vessel_type", len(arrays["vessel_type_names"]))
        ds.createDimension("article_class", len(arrays["article_class_names"]))

        t = _variable(ds, "time", ("time",), [datetime.fromisoformat(config["time_utc"].replace("Z", "+00:00")).timestamp()],
                      "seconds since 1970-01-01 00:00:00 UTC")
        t.calendar = "standard"
        _variable(ds, "frequency_hz", ("frequency",), config["frequency_hz"], "Hz")
        _variable(ds, "source_cell_id", ("source_cell",), np.arange(nsource), "1")
        _variable(ds, "lon", ("source_cell",), arrays["lon"], "degrees_east", "GFW grid-cell center longitude")
        _variable(ds, "lat", ("source_cell",), arrays["lat"], "degrees_north", "GFW grid-cell center latitude")
        _variable(ds, "cell_area_m2", ("source_cell",), arrays["cell_area_m2"], "m2")
        _variable(ds, "source_depth_m", ("source_cell",), np.full(nsource, config["source_depth_m"]), "m")
        for name in ("presence_record_count", "included_vessel_count", "proxy_vessel_count", "excluded_record_count"):
            _variable(ds, name, ("source_cell",), arrays[name], "1")
        _variable(ds, "vessel_hours", ("source_cell",), arrays["vessel_hours"], "h")
        _variable(ds, "included_vessel_hours", ("source_cell",), arrays["included_vessel_hours"], "h")
        _variable(ds, "vessel_type_count", ("source_cell", "vessel_type"), arrays["vessel_type_count"], "1")
        _variable(ds, "article_class_hours", ("source_cell", "article_class"), arrays["article_class_hours"], "h")
        _variable(ds, "article_class_mean_msl_db", ("article_class", "frequency"), arrays["article_class_mean_db"], "dB re 1 uPa m")
        _variable(ds, "article_class_sd_db", ("article_class", "frequency"), arrays["article_class_sd_db"], "dB")
        _variable(ds, "cell_source_power_linear", ("frequency", "source_cell"), arrays["cell_source_power_linear"], "uPa2 m2")
        _variable(ds, "cell_source_level_db", ("frequency", "source_cell"), arrays["cell_source_level_db"], "dB re 1 uPa m")
        low = _variable(ds, "cell_source_level_minus_1sd_db", ("frequency", "source_cell"), arrays["cell_source_level_minus_1sd_db"], "dB re 1 uPa m")
        low.long_name = "sensitivity with every article class shifted down by its reported 1-sigma SD"
        high = _variable(ds, "cell_source_level_plus_1sd_db", ("frequency", "source_cell"), arrays["cell_source_level_plus_1sd_db"], "dB re 1 uPa m")
        high.long_name = "sensitivity with every article class shifted up by its reported 1-sigma SD"
        names = ds.createVariable("vessel_type_name", str, ("vessel_type",))
        names[:] = np.asarray([x or "<blank>" for x in arrays["vessel_type_names"]], dtype=object)
        class_names = ds.createVariable("article_class_name", str, ("article_class",))
        class_names[:] = np.asarray(arrays["article_class_names"], dtype=object)


def build_shipping_source_product(output_path, config, input_path):
    """从一个小时的 GFW presence 文件原子生成可缓存的船舶源网格。"""
    output_path = Path(output_path)
    input_path = Path(input_path)
    validate_config(config)
    arrays, audit = read_presence(input_path, config)
    checksum = file_sha256(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        export_netcdf(temporary, arrays, audit, config, input_path, checksum)
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "output_file": str(output_path.resolve()),
        "time_utc": str(config["time_utc"]),
        "input_file": str(input_path.resolve()),
        "input_sha256": checksum,
        "active_acoustic_cells": audit["active_acoustic_cells"],
    }


def build_summary(arrays, audit, config, input_path, checksum, output_dir):
    active = arrays["included_vessel_count"] > 0
    levels = arrays["cell_source_level_db"][:, active]
    top_index = np.argsort(arrays["cell_source_level_db"][0])[::-1][:10]
    summary = {
        "model_version": MODEL_VERSION,
        "model_status": "uncalibrated_global_engineering_source_scenario",
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": checksum,
        "output_dir": str(Path(output_dir).resolve()),
        "config": config,
        "audit": audit,
        "active_cell_level_statistics_db": {
            str(int(f)): {
                "min": float(np.min(levels[i])),
                "median": float(np.median(levels[i])),
                "max": float(np.max(levels[i])),
            } for i, f in enumerate(config["frequency_hz"])
        },
        "top_63hz_cells": [
            {
                "lon": float(arrays["lon"][i]),
                "lat": float(arrays["lat"][i]),
                "included_vessels": int(arrays["included_vessel_count"][i]),
                "proxy_vessels": int(arrays["proxy_vessel_count"][i]),
                "level_63hz_db": float(arrays["cell_source_level_db"][0, i]),
                "level_1000hz_db": float(arrays["cell_source_level_db"][1, i]),
            } for i in top_index if arrays["included_vessel_count"][i] > 0
        ],
        "warnings": [
            "源级采用文章在 Eclipse Sound 的船型平均观测值，不能视作全球校准值。",
            "GFW 数据缺少船速、船长和连续航迹，无法复现文章的静止船过滤及逐分钟轨迹处理。",
            "FISHING、OTHER、CARRIER、BUNKER、SEISMIC_VESSEL 和 SUPPORT 使用显式代理映射。",
            "SEISMIC_VESSEL 仅表示推进噪声代理，不含气枪等脉冲源。",
            "GEAR、DISCREPANCY 和缺失船型不计入声功率，但保留在存在记录统计中。",
            "结果只表示源网格；未计算传播损失、海冰影响或接收声级。"
        ],
    }
    return summary


def export_plot(path, arrays, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = arrays["included_vessel_count"] > 0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    for fi, (axis, frequency) in enumerate(zip(axes, config["frequency_hz"])):
        values = arrays["cell_source_level_db"][fi, active]
        vmin, vmax = np.percentile(values, [2, 98])
        points = axis.scatter(arrays["lon"][active], arrays["lat"][active], c=values,
                              s=2.0, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xticks(np.arange(-180, 181, 60))
        axis.set_yticks(np.arange(-90, 91, 30))
        axis.grid(True, color="#D9D9D9", linewidth=0.4)
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
        axis.set_title(f"{frequency:g} Hz cell source level (article-class proxy)")
        bar = fig.colorbar(points, ax=axis, pad=0.01)
        bar.set_label("dB re 1 uPa m")
    fig.suptitle("Global distributed shipping-noise source grid, 2025-01-02 12:00 UTC", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path, summary):
    a = summary["audit"]
    stats = summary["active_cell_level_statistics_db"]
    mapping_rows = []
    for vessel_type, item in summary["config"]["gfw_vessel_type_mapping"].items():
        label = vessel_type or "<blank>"
        mapping_rows.append(f"| {label} | {item['article_class'] or 'excluded'} | {item['mapping']} |")
    text = f"""# 全球船舶分布式噪声源网格（首版）

时刻：{summary['config']['time_utc']}。输入采用 GFW LOW 0.1° 船舶存在网格；输出采用非空单元稀疏存储。

## 结果

- 输入记录：{a['input_rows']:,}；唯一船舶：{a['unique_vessel_ids']:,}。
- 非空存在单元：{a['occupied_presence_cells']:,}；含已建模船舶的声源单元：{a['active_acoustic_cells']:,}。
- 计入声功率：{a['included_records']:,} 条；其中代理映射：{a['proxy_records']:,} 条。
- 排除：{a['excluded_records']:,} 条，主要为 GEAR 或未解析船型。
- 63 Hz 活跃单元源级：{stats['63']['min']:.1f}–{stats['63']['max']:.1f} dB，中位数 {stats['63']['median']:.1f} dB re 1 μPa·m。
- 1000 Hz 活跃单元源级：{stats['1000']['min']:.1f}–{stats['1000']['max']:.1f} dB，中位数 {stats['1000']['median']:.1f} dB re 1 μPa·m。

## 方法

参考 Johnson 等（2026）的船舶噪声建模框架，使用 63 Hz 与 1000 Hz、5 m 源深度、时长权重和线性功率非相干叠加。每个 GFW 单元的小时平均等效源功率为：

`P_cell(f) = Σ_class H_class × 10^(MSL_class(f)/10)`

由于本次平均时段为 1 h，`H_class` 以船舶存在小时计。`cell_source_level_db = 10 log10(P_cell)`。

文章报告的船型平均 MSL：货船 157.8/136.1 dB、油轮 170.1/144.7 dB、破冰船 151.0/144.4 dB、客船 147.0/132.4 dB，分别对应 63/1000 Hz。文章来源：{summary['config']['article_reference']['journal']}，DOI {summary['config']['article_reference']['doi']}。

## 船型映射

| GFW 船型 | 文章船型 | 状态 |
|---|---|---|
{chr(10).join(mapping_rows)}

## 输出文件

- `shipping_sources.nc`：稀疏全球源网格、分类数量、线性源功率、dB 源级和 ±1 个文章船型标准差敏感性结果。
- `summary.json`：输入校验、分类统计、极值单元和全部假设。
- `source_grid_overview.png`：63 Hz 与 1000 Hz 全球源网格预览。
- `config_used.json`：实际参数与文献映射。

## 限制

{chr(10).join('- ' + item for item in summary['warnings'])}

该结果不能直接解释为接收点噪声级。下一阶段需针对每个源单元计算与 5 m 单极源约定一致的传播损失，再在线性域合成接收声场。
"""
    Path(path).write_text(text, encoding="utf-8")


def run(input_path, config_path, output_dir):
    input_path = Path(input_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖: {output}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    validate_config(config)
    arrays, audit = read_presence(input_path, config)
    checksum = file_sha256(input_path)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(arrays, audit, config, input_path, checksum, output)
    export_netcdf(output / "shipping_sources.nc", arrays, audit, config, input_path, checksum)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    export_plot(output / "source_grid_overview.png", arrays, config)
    write_report(output / "model_report.md", summary)
    print(json.dumps({
        "output_dir": str(output.resolve()),
        "input_rows": audit["input_rows"],
        "occupied_presence_cells": audit["occupied_presence_cells"],
        "active_acoustic_cells": audit["active_acoustic_cells"],
        "included_records": audit["included_records"],
        "excluded_records": audit["excluded_records"],
    }, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成 GFW 全球船舶分布式噪声源网格。")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "test_api" / "gfw_output_fixed" / "20250102_12UTC" / "LOW" / "aggregate" / "global" / "presence_20250102_12UTC.csv" / "presence_20250102_12UTC.csv"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "jobs" / "shipping_sources_gfw_20250102_12utc_v1"))
    args = parser.parse_args()
    run(args.input, args.config, args.output_dir)


if __name__ == "__main__":
    main()
