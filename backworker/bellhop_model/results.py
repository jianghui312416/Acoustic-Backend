# -*- coding: utf-8 -*-
"""用于 BellhopCXX 的 .ray/.shd 解析，以及拼接后三维结果导出。

本模块把 Bellhop 原生文件转换成网页应用使用的五个公开 NetCDF 产品，
并生成交互式结果页面读取的紧凑 JSON 缓存。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator


RAY_COLORS = {
    "none": [1.0, 0.0, 0.0],
    "surface_only": [0.0, 1.0, 0.0],
    "bottom_only": [0.0, 0.0, 1.0],
    "surface_and_bottom": [0.0, 0.0, 0.0],
}
SOURCE_COLORS = [
    [0.88, 0.08, 0.08], [1.0, 0.42, 0.0], [0.72, 0.05, 0.72],
    [0.0, 0.55, 0.72], [0.18, 0.62, 0.24], [0.45, 0.28, 0.82],
]


# =============================================================================
# 通用转换和 JSON 工具
# =============================================================================
def _finite_float(value: Any, default: float = 0.0) -> float:
    """把输入转换为有限浮点数，失败时返回指定默认值。"""
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _json_array(values, decimals=None):
    """返回 JSON 安全列表，并把 NaN/Inf 转成 null。"""
    arr = np.asarray(values)
    if decimals is not None and np.issubdtype(arr.dtype, np.number):
        arr = np.round(arr.astype(np.float64, copy=False), decimals)
    if not np.issubdtype(arr.dtype, np.number):
        return arr.tolist()
    obj = arr.astype(object)
    obj[~np.isfinite(arr.astype(np.float64, copy=False))] = None
    return obj.tolist()


def _write_json(path, value):
    """以 UTF-8 编码和缩进格式写入 JSON 文件。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def _reflection_style(ntop: int, nbot: int) -> Tuple[List[float], str]:
    """根据海面和海底反射次数返回前端线型及分类名称。"""
    if ntop >= 1 and nbot >= 1:
        kind = "surface_and_bottom"
    elif nbot >= 1:
        kind = "bottom_only"
    elif ntop >= 1:
        kind = "surface_only"
    else:
        kind = "none"
    return RAY_COLORS[kind], kind


# =============================================================================
# 声线文件读取（Bellhop）
# =============================================================================
def parse_ray_file(filepath):
    """解析 Bellhop/BellhopCXX 格式化声线文件，不绘图。"""
    text = Path(filepath).read_text(encoding="ascii", errors="replace")
    tokens = text.split()
    pos = 0

    def next_token() -> str:
        """从声线文件词元流中读取下一个非空词元。"""
        nonlocal pos
        if pos >= len(tokens):
            raise EOFError("ray 文件意外结束")
        value = tokens[pos]
        pos += 1
        if value.startswith(("'", '"')):
            quote = value[0]
            while not value.endswith(quote) or len(value) == 1:
                if pos >= len(tokens):
                    break
                value += " " + tokens[pos]
                pos += 1
            value = value.strip("'\"")
        return value

    def next_float() -> float:
        """读取下一个词元并转换为浮点数。"""
        return float(next_token())

    def next_int() -> int:
        """读取下一个词元并转换为整数。"""
        return int(float(next_token()))

    # 二维和三维声线文件头在 Bellhop 中不同。先检查第三个物理行
    # （声源维度行），再进入 token 流解析，这样两种格式可以共用下面的声线主体解析。
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    source_dimension_count = 3
    if len(nonempty_lines) >= 3:
        source_dimension_count = len(nonempty_lines[2].split())

    try:
        title = next_token()
        freq = next_float()
        if source_dimension_count == 1:
            # 二维：NSD、Nalpha、顶部深度、底部深度、坐标类型。
            nsx, nsy, nsz = 1, 1, next_int()
            nalpha, nbeta = next_int(), 1
        else:
            # 三维/Nx2D：NSX、NSY、NSZ，然后是 Nalpha 和 Nbeta。
            nsx, nsy, nsz = next_int(), next_int(), next_int()
            nalpha, nbeta = next_int(), next_int()
        depth_top, depth_bottom = next_float(), next_float()
        coord_type = next_token().strip().lower()
    except (EOFError, ValueError) as exc:
        raise ValueError(f"无效的 Bellhop ray 文件头：{filepath}") from exc

    is_xyz = "xyz" in coord_type
    expected = nsx * nsy * nsz * nalpha * nbeta
    rays: List[Dict[str, Any]] = []
    for ray_index in range(expected):
        try:
            alpha0 = next_float()
            nsteps, ntop, nbot = next_int(), next_int(), next_int()
            if nsteps < 1:
                continue
            count = 3 if is_xyz else 2
            values = np.fromiter((next_float() for _ in range(nsteps * count)), dtype=np.float64)
            points = values.reshape(nsteps, count)
        except (EOFError, ValueError):
            break
        color, reflection = _reflection_style(ntop, nbot)
        rays.append({
            "id": ray_index + 1,
            "alpha0": float(alpha0),
            "nsteps": int(nsteps),
            "ntop": int(ntop),
            "nbot": int(nbot),
            "points": points,
            "color": color,
            "reflection_type": reflection,
        })

    if not rays:
        raise ValueError(f"Bellhop 未在 {filepath} 中生成有效声线")
    return {
        "title": title,
        "freq": freq,
        "nsx": nsx,
        "nsy": nsy,
        "nsz": nsz,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "depth_top": depth_top,
        "depth_bottom": depth_bottom,
        "coord_type": coord_type,
        "expected_ray_count": expected,
        "actual_ray_count": len(rays),
        "simulation_dimension": "2D" if source_dimension_count == 1 or not is_xyz else "3D",
    }, rays


# =============================================================================
# 经纬度和本地坐标转换
# =============================================================================
def local_offset_to_lonlat(x_km, y_km, source_lon, source_lat):
    """按参考程序的 WGS84 半径，把本地东西/南北偏移转换为经纬度。"""
    y = np.asarray(y_km, dtype=np.float64)
    x = np.asarray(x_km, dtype=np.float64)
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    lat0_rad = math.radians(float(source_lat))
    meridional = semi_major * (1.0 - eccentricity_sq) / (1.0 - eccentricity_sq * math.sin(lat0_rad) ** 2) ** 1.5
    prime_vertical = semi_major / math.sqrt(1.0 - eccentricity_sq * math.sin(lat0_rad) ** 2)
    east_radius = prime_vertical * math.cos(lat0_rad)
    if abs(east_radius) < 1.0:
        east_radius = math.copysign(1.0, east_radius if east_radius else 1.0)
    lat = source_lat + np.rad2deg(y * 1000.0 / meridional)
    lon = source_lon + np.rad2deg(x * 1000.0 / east_radius)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


def lonlat_to_local_offset(lon, lat, origin_lon, origin_lat):
    """函数 local_offset_to_lonlat 的反向转换，以单个 WGS84 切点为原点。"""
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    lat0_rad = math.radians(float(origin_lat))
    meridional = semi_major * (1.0 - eccentricity_sq) / (1.0 - eccentricity_sq * math.sin(lat0_rad) ** 2) ** 1.5
    prime_vertical = semi_major / math.sqrt(1.0 - eccentricity_sq * math.sin(lat0_rad) ** 2)
    east_radius = prime_vertical * math.cos(lat0_rad)
    if abs(east_radius) < 1.0:
        east_radius = math.copysign(1.0, east_radius if east_radius else 1.0)
    lon_values = np.asarray(lon, dtype=np.float64)
    lat_values = np.asarray(lat, dtype=np.float64)
    delta_lon = (lon_values - float(origin_lon) + 180.0) % 360.0 - 180.0
    x = np.deg2rad(delta_lon) * east_radius / 1000.0
    y = np.deg2rad(lat_values - float(origin_lat)) * meridional / 1000.0
    return x, y


def normalise_rays(
    rays,
    header,
    source_lon,
    source_lat,
    source_x_grid_km,
    source_y_grid_km,
    calculation_radius_km,
    origin_lon=None,
    origin_lat=None,
):
    """把 Bellhop 绝对网格坐标转换为相对声源的公里坐标。"""
    converted: List[Dict[str, Any]] = []
    is_xyz = "xyz" in str(header.get("coord_type", "")).lower()
    for item in rays:
        points = np.asarray(item["points"], dtype=np.float64)
        if is_xyz:
            x_absolute = points[:, 0] / 1000.0
            y_absolute = points[:, 1] / 1000.0
            x_rel = x_absolute - source_x_grid_km
            y_rel = y_absolute - source_y_grid_km
            depth = np.abs(points[:, 2])
        else:
            # 坐标格式 RZ 只是兼容兜底；三维网页任务正常会输出 XYZ。
            radial = points[:, 0] / 1000.0
            bearing = math.radians(_finite_float(item.get("bearing_deg"), 0.0))
            # 方位角从地理北向开始顺时针增加；本地 x/y 分别表示东向/北向。
            x_rel = radial * math.sin(bearing)
            y_rel = radial * math.cos(bearing)
            depth = np.abs(points[:, 1])
        good = np.isfinite(x_rel) & np.isfinite(y_rel) & np.isfinite(depth)
        good &= np.hypot(x_rel, y_rel) <= calculation_radius_km * 1.03 + 0.2
        if np.count_nonzero(good) < 2:
            continue
        x_rel, y_rel, depth = x_rel[good], y_rel[good], depth[good]
        if is_xyz and origin_lon is not None and origin_lat is not None:
            lon, lat = local_offset_to_lonlat(x_absolute[good], y_absolute[good], origin_lon, origin_lat)
        else:
            lon, lat = local_offset_to_lonlat(x_rel, y_rel, source_lon, source_lat)
        dx = np.diff(x_rel)
        dy = np.diff(y_rel)
        dz = np.diff(depth) / 1000.0
        seg = np.r_[0.0, np.sqrt(dx * dx + dy * dy + dz * dz)]
        out = dict(item)
        out.update({
            "x_km": x_rel,
            "y_km": y_rel,
            "depth_m": depth,
            "lon": lon,
            "lat": lat,
            "segment_length_km": seg,
            "path_distance_km": np.cumsum(seg),
        })
        converted.append(out)
    if not converted:
        raise ValueError("Bellhop 声线在坐标转换和计算半径裁剪后为空")
    return converted



# =============================================================================
# 声线反射分类
# =============================================================================
def _clustered_true_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    """返回连续 True 区段的闭区间索引范围。"""
    values = np.asarray(mask, dtype=bool).ravel()
    ranges: List[Tuple[int, int]] = []
    start = -1
    for index in range(values.size + 1):
        active = index < values.size and bool(values[index])
        if active and start < 0:
            start = index
        elif not active and start >= 0:
            ranges.append((start, index - 1))
            start = -1
    return ranges


def reclassify_ray_reflections(
    rays: Sequence[Dict[str, Any]],
    x_axis_km: np.ndarray,
    y_axis_km: np.ndarray,
    bottom_depth_m: np.ndarray,
    surface_tolerance_m: float = 3.0,
    bottom_tolerance_m: float = 8.0,
) -> List[Dict[str, Any]]:
    """用声线几何和局部海底地形补充 Bellhop 反射次数。

    Bellhop 格式化声线文件通常会给出 NumTopBnc 和 NumBotBnc。
    这里的几何检查故意偏保守，只用于补充反射次数，不会减少 Bellhop 已给出的计数。
    它也能处理反射位置附近常见的重复边界点。
    """
    interp = RegularGridInterpolator(
        (np.asarray(y_axis_km, dtype=np.float64), np.asarray(x_axis_km, dtype=np.float64)),
        np.asarray(bottom_depth_m, dtype=np.float64),
        bounds_error=False,
        fill_value=np.nan,
    )
    updated: List[Dict[str, Any]] = []
    for original in rays:
        ray = dict(original)
        x = np.asarray(ray.get("x_km", []), dtype=np.float64)
        y = np.asarray(ray.get("y_km", []), dtype=np.float64)
        depth = np.asarray(ray.get("depth_m", []), dtype=np.float64)
        n = min(x.size, y.size, depth.size)
        if n < 3:
            updated.append(ray)
            continue
        x, y, depth = x[:n], y[:n], depth[:n]
        bottom = interp(np.column_stack([y, x]))

        top_near = np.isfinite(depth) & (depth <= max(0.2, float(surface_tolerance_m)))
        bottom_clearance = bottom - depth
        bottom_near = (
            np.isfinite(depth) & np.isfinite(bottom_clearance)
            & (bottom_clearance >= -max(1.0, float(bottom_tolerance_m)))
            & (bottom_clearance <= max(1.0, float(bottom_tolerance_m)))
        )

        def count_turning_clusters(mask: np.ndarray, is_bottom: bool) -> int:
            """统计靠近指定边界且发生转向的连续点簇数量。"""
            count = 0
            for first, last in _clustered_true_ranges(mask):
                if first <= 0 or last >= n - 1:
                    continue
                left = first - 1
                right = last + 1
                cluster = depth[first:last + 1]
                if not np.any(np.isfinite(cluster)):
                    continue
                reference = float(np.nanmax(cluster) if is_bottom else np.nanmin(cluster))
                # 真正的反射会先靠近边界，然后离开边界。
                if is_bottom:
                    turned = depth[left] < reference - 0.05 and depth[right] < reference - 0.05
                else:
                    turned = depth[left] > reference + 0.05 and depth[right] > reference + 0.05
                if turned:
                    count += 1
            return count

        inferred_top = count_turning_clusters(top_near, False)
        inferred_bottom = count_turning_clusters(bottom_near, True)
        ntop = max(int(ray.get("ntop", 0) or 0), inferred_top)
        nbot = max(int(ray.get("nbot", 0) or 0), inferred_bottom)
        color, reflection = _reflection_style(ntop, nbot)
        ray.update({
            "ntop": ntop,
            "nbot": nbot,
            "inferred_ntop": inferred_top,
            "inferred_nbot": inferred_bottom,
            "color": color,
            "reflection_type": reflection,
        })
        updated.append(ray)
    return updated


# =============================================================================
# 传播损失 SHD 读取整理
# =============================================================================
def align_shd_axes_to_environment(
    shd: Dict[str, Any],
    beta_min_deg: float,
    beta_max_deg: float,
    range_max_km: float,
    receiver_depth_max_m: float,
    receiver_range_min_km: float = 0.0,
    receiver_depth_min_m: float = 0.0,
) -> Dict[str, Any]:
    """用写入 .env 的精确坐标轴替换 SHD 中的坐标向量。

    不同 BellhopCXX/Acoustics Toolbox 版本写坐标记录时可能使用不同二进制精度。
    本后端写出的接收坐标轴本来就是均匀向量，因此以提交给 Bellhop 的向量为准，
    可以在不改变任何声压采样值的情况下重建坐标轴。
    """
    out = dict(shd)
    pressure_tl = np.asarray(out["tl"])
    if pressure_tl.ndim != 4:
        raise ValueError("SHD TL 数组维度无效")
    ntheta, _, ndepth, nrange = pressure_tl.shape
    theta = np.linspace(float(beta_min_deg), float(beta_max_deg), ntheta, dtype=np.float64)
    depth = np.linspace(
        max(0.0, float(receiver_depth_min_m)),
        max(0.0, float(receiver_depth_max_m)),
        ndepth,
        dtype=np.float64,
    )
    ranges = np.linspace(
        max(0.0, float(receiver_range_min_km)),
        max(0.0, float(range_max_km)),
        nrange,
        dtype=np.float64,
    )
    out["theta_raw"] = np.asarray(out.get("theta_deg", []), dtype=np.float64)
    out["receiver_depth_raw"] = np.asarray(out.get("receiver_depth_m", []), dtype=np.float64)
    out["receiver_range_raw"] = np.asarray(out.get("range_km", []), dtype=np.float64)
    out["theta_deg"] = theta
    out["receiver_depth_m"] = depth
    out["range_km"] = ranges
    out["axis_source"] = "environment_request"
    return out

def _position_vector_score(values: np.ndarray, kind: str) -> float:
    """给一个 SHD 坐标向量候选结果打分，用于判断二进制精度。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return -1e9
    finite = np.isfinite(arr)
    score = 8.0 * float(np.mean(finite))
    if not np.all(finite):
        return score - 30.0

    absmax = float(np.max(np.abs(arr)))
    if kind == "theta":
        score += 5.0 if absmax <= 720.0 else -12.0
    elif kind in {"source_depth", "receiver_depth"}:
        score += 5.0 if float(np.min(arr)) >= -10.0 and absmax <= 1.0e6 else -12.0
    elif kind == "range":
        score += 5.0 if float(np.min(arr)) >= -10.0 and absmax <= 1.0e9 else -12.0
    else:
        score += 2.0 if absmax <= 1.0e9 else -8.0

    if arr.size >= 2:
        delta = np.diff(arr)
        tolerance = max(1e-9, 1e-8 * max(1.0, absmax))
        nondecreasing = float(np.mean(delta >= -tolerance))
        score += 12.0 * nondecreasing
        # 如果把 float64 向量错当成 float32 读取，常会变成
        # [低位字, 高位字, 低位字, 高位字, ...]，
        # 造成大量 0/小数交替，以及明显不单调的坐标轴。
        if arr.size >= 4:
            zero_fraction = float(np.mean(np.isclose(arr, 0.0, atol=tolerance)))
            sign_changes = float(np.mean(delta[:-1] * delta[1:] < 0.0)) if delta.size > 1 else 0.0
            score -= 5.0 * max(0.0, zero_fraction - 0.45)
            score -= 8.0 * sign_changes
    return score


def _detect_shd_position_dtype(
    handle: Any,
    record_bytes: int,
    counts: Dict[int, int],
) -> str:
    """判断 SHD 坐标记录使用的是 IEEE float32 还是 float64。"""
    kinds = {
        4: "theta", 5: "source_x", 6: "source_y",
        7: "source_depth", 8: "receiver_depth", 9: "range",
    }
    totals: Dict[str, float] = {"<f4": 0.0, "<f8": 0.0}
    usable: Dict[str, int] = {"<f4": 0, "<f8": 0}
    for dtype in ("<f4", "<f8"):
        itemsize = np.dtype(dtype).itemsize
        for record, count in counts.items():
            if count <= 0 or count * itemsize > record_bytes:
                totals[dtype] -= 100.0
                continue
            handle.seek(record * record_bytes)
            raw = handle.read(count * itemsize)
            if len(raw) != count * itemsize:
                totals[dtype] -= 100.0
                continue
            values = np.frombuffer(raw, dtype=dtype, count=count)
            # 单元素声源坐标很难区分精度；theta/depth/range 向量权重更高。
            weight = 0.25 if count == 1 else 1.0
            totals[dtype] += weight * _position_vector_score(values, kinds[record])
            usable[dtype] += 1
    if usable["<f8"] == 0:
        return "<f4"
    # 分数几乎相同的时候，优先使用历史 float32 布局。
    return "<f8" if totals["<f8"] > totals["<f4"] + 1.0 else "<f4"


def read_shd_bin(filepath: os.PathLike[str] | str) -> Dict[str, Any]:
    """读取 Acoustics Toolbox/BellhopCXX 直接访问格式 .shd 文件。

    随附的 BellhopCXX 可能把坐标向量写成 32 位或 64 位实数，
    而声压采样仍是 complex float32。这里根据真实坐标记录自动判断精度，
    避免硬编码精度导致方位角或距离轴出现加倍、交错等假数据。
    """
    path = Path(filepath)
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(4)
        if len(raw) != 4:
            raise ValueError("SHD 文件为空")
        recl_words = struct.unpack("<i", raw)[0]
        record_bytes = 4 * recl_words
        if recl_words <= 0 or record_bytes > size:
            raise ValueError("SHD 记录长度无效")
        if size % record_bytes != 0:
            raise ValueError("SHD 文件长度不是记录长度的整数倍")

        title = handle.read(80).split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        handle.seek(record_bytes)
        plot_record = handle.read(min(record_bytes, 16))
        # 当前文件仍占用 10 字节 PlotType 记录；读取前 4 字节也能兼容
        # 把它暴露成 fourcc 整数的工具。
        plot_type = plot_record[:10].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        if not plot_type:
            plot_type = plot_record[:4].decode("ascii", errors="replace").strip("\x00 ")
        plot_lower = plot_type.lower()
        is_tl = plot_lower.startswith("tl")
        irregular = plot_lower.startswith("irre")
        if not (is_tl or irregular or plot_lower.startswith("rect")):
            raise ValueError(f"SHD PlotType 无法识别：{plot_type!r}")

        handle.seek(2 * record_bytes)
        dims = handle.read(7 * 4)
        if len(dims) != 28:
            raise ValueError("SHD 维度记录不完整")
        nfreq, ntheta, nsx, nsy, nsz, nrd, nrr = struct.unpack("<7i", dims)
        if min(nfreq, ntheta, nsx, nsy, nsz, nrd, nrr) <= 0:
            raise ValueError("SHD 维度包含非正数")
        if nsx != 1 or nsy != 1:
            raise ValueError("当前网页解析器仅支持每个 SHD 文件一个水平声源位置")

        sx_count = 2 if is_tl else nsx
        sy_count = 2 if is_tl else nsy
        position_counts = {
            4: ntheta, 5: sx_count, 6: sy_count,
            7: nsz, 8: nrd, 9: nrr,
        }
        position_dtype = _detect_shd_position_dtype(handle, record_bytes, position_counts)
        position_itemsize = np.dtype(position_dtype).itemsize

        def read_at(record: int, count: int, dtype: str) -> np.ndarray:
            """从指定直接访问记录读取固定数量的数组元素。"""
            itemsize = np.dtype(dtype).itemsize
            if count * itemsize > record_bytes:
                raise ValueError(
                    f"SHD 第 {record + 1} 条记录容量不足：需要 {count * itemsize} 字节，"
                    f"记录仅有 {record_bytes} 字节"
                )
            handle.seek(record * record_bytes)
            raw_values = handle.read(count * itemsize)
            if len(raw_values) != count * itemsize:
                raise ValueError(f"SHD 第 {record + 1} 条记录不完整")
            return np.frombuffer(raw_values, dtype=dtype, count=count).copy()

        # 在 AT 兼容布局和 BellhopCXX 布局中，频率向量都是双精度。
        # 坐标向量精度已经在上面自动判断。
        freq = read_at(3, nfreq, "<f8")
        theta = read_at(4, ntheta, position_dtype).astype(np.float64)
        sx = read_at(5, sx_count, position_dtype).astype(np.float64)
        sy = read_at(6, sy_count, position_dtype).astype(np.float64)
        if sx_count == 2 and nsx != 2:
            sx = np.linspace(sx[0], sx[-1], nsx)
        if sy_count == 2 and nsy != 2:
            sy = np.linspace(sy[0], sy[-1], nsy)
        source_depth = read_at(7, nsz, position_dtype).astype(np.float64)
        receiver_depth = read_at(8, nrd, position_dtype).astype(np.float64)
        # 距离 range 记录可能单独使用不同精度（bellhopcxx 的 range 是 float64，
        # 而 depth 仍是 float32），因此对 range 单独做一次精度检测。
        range_dtype = _detect_shd_position_dtype(handle, record_bytes, {9: nrr})
        receiver_range = read_at(9, nrr, range_dtype).astype(np.float64)

        # 第 2 条记录包含七个 int32 维度，后面跟着名义频率和衰减值；
        # 这些实数使用当前构建的实数精度。这里它们只作为信息，
        # 真正的频率以第 3 条记录为准。
        header_real_count = 2
        header_reals = np.array([], dtype=np.float64)
        if 28 + header_real_count * position_itemsize <= record_bytes:
            handle.seek(2 * record_bytes + 28)
            header_raw = handle.read(header_real_count * position_itemsize)
            if len(header_raw) == header_real_count * position_itemsize:
                header_reals = np.frombuffer(header_raw, dtype=position_dtype).astype(np.float64)
        atten = float(header_reals[1]) if header_reals.size >= 2 else math.nan

        receivers_per_range = 1 if irregular else nrd
        needed_records = 10 + ntheta * nsz * receivers_per_range
        if size < needed_records * record_bytes:
            raise ValueError(
                f"SHD 只有 {size} 字节，压力记录不完整；请检查 Bellhop .prt 中的 FATAL ERROR"
            )
        if 2 * nrr * np.dtype("<f4").itemsize > record_bytes:
            raise ValueError("SHD 压力记录长度不足，无法容纳全部复声压样本")

        pressure = np.full((ntheta, nsz, nrd, nrr), np.nan + 1j * np.nan, dtype=np.complex64)
        for itheta in range(ntheta):
            for isz in range(nsz):
                for ird in range(receivers_per_range):
                    recnum = 10 + itheta * nsz * receivers_per_range + isz * receivers_per_range + ird
                    values = read_at(recnum, 2 * nrr, "<f4")
                    pressure[itheta, isz, ird, :] = values[0::2] + 1j * values[1::2]

    magnitude = np.abs(pressure).astype(np.float64)
    magnitude[(~np.isfinite(magnitude)) | (magnitude <= 0.0) | (magnitude > 1e20)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = -20.0 * np.log10(magnitude)
    if not np.any(np.isfinite(tl)):
        raise ValueError("SHD 压力场没有有限值；Bellhop TL 计算未成功")
    range_km = receiver_range / 1000.0 if np.nanmax(np.abs(receiver_range)) > 1000.0 else receiver_range
    return {
        "title": title,
        "plot_type": plot_type,
        "freq_hz": float(freq[0]),
        "atten": atten if math.isfinite(atten) else math.nan,
        "theta_deg": theta,
        "source_x_m": sx,
        "source_y_m": sy,
        "source_depth_m": source_depth,
        "receiver_depth_m": receiver_depth,
        "range_km": range_km,
        # 保留 Bellhop 原生复声压，供垂直阵互谱密度和相干处理使用。
        # 常规 TL 导出仍使用下面的 ``tl``，既有调用不受影响。
        "pressure": pressure,
        "tl": tl,
        "record_words": recl_words,
        "position_dtype": "float64" if position_dtype == "<f8" else "float32",
    }


# =============================================================================
# 地形采样和声线 TL 附加
# =============================================================================
def sample_bottom_polar(
    x_axis_km: np.ndarray,
    y_axis_km: np.ndarray,
    bottom_depth_m: np.ndarray,
    theta_deg: np.ndarray,
    range_km: np.ndarray,
) -> np.ndarray:
    """在 theta/range 声线路径上采样规则的声源相对海底网格。"""
    interp = RegularGridInterpolator(
        (np.asarray(y_axis_km), np.asarray(x_axis_km)),
        np.asarray(bottom_depth_m),
        bounds_error=False,
        fill_value=np.nan,
    )
    theta_rad = np.deg2rad(theta_deg)[:, None]
    xx = np.sin(theta_rad) * range_km[None, :]
    yy = np.cos(theta_rad) * range_km[None, :]
    return interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(len(theta_deg), len(range_km))


def _nearest_sorted_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """返回最近邻索引，避免分配 len(axis) × len(values) 的大数组。"""
    coordinates = np.asarray(axis, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    if coordinates.size == 1:
        return np.zeros(query.shape, dtype=np.int64)
    order = np.argsort(coordinates)
    ordered = coordinates[order]
    position = np.searchsorted(ordered, query)
    position = np.clip(position, 1, len(ordered) - 1)
    left = ordered[position - 1]
    right = ordered[position]
    choose_right = np.abs(query - right) < np.abs(query - left)
    return order[position - 1 + choose_right.astype(np.int64)]


def attach_tl_to_rays(source: Dict[str, Any]) -> None:
    """为每个声线点附加最近邻 TL，供 NetCDF 导出使用。"""
    model = source["tl"]
    theta = np.asarray(model["theta_deg"])
    depth = np.asarray(model["depth_m"])
    ranges = np.asarray(model["range_km"])
    layers = np.asarray(model["layers"])
    for ray in source["rays"]:
        x = np.asarray(ray["x_km"])
        y = np.asarray(ray["y_km"])
        z = np.asarray(ray["depth_m"])
        rr = np.hypot(x, y)
        bearings = (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0
        it = np.argmin(np.abs(((theta[:, None] - bearings[None, :] + 180.0) % 360.0) - 180.0), axis=0)
        iz = _nearest_sorted_indices(depth, z)
        ir = _nearest_sorted_indices(ranges, rr)
        values = layers[iz, it, ir]
        values[rr > _finite_float(source.get("calculation_radius_km"), ranges[-1]) * 1.01] = np.nan
        ray["tl_db"] = values


# =============================================================================
# 文件写入工具（NetCDF）
# =============================================================================
def _nc_variable(ds: netCDF4.Dataset, name: str, dtype: str, dims: tuple, **attrs: Any):
    """创建带压缩和属性的 NetCDF 变量。"""
    fill = attrs.pop("fill_value", None)
    kwargs = {"zlib": True, "complevel": 4, "shuffle": True}
    if fill is not None:
        kwargs["fill_value"] = fill
    var = ds.createVariable(name, dtype, dims, **kwargs)
    for key, value in attrs.items():
        setattr(var, key, value)
    return var


@contextmanager
def _netcdf_writer(filename: os.PathLike[str] | str):
    """即使 Windows 路径包含非 ASCII 字符，也安全创建 NetCDF 文件。

    某些 Windows HDF5/netCDF4 构建会把 Unicode 路径传给窄字符编码 API。
    这会让包含中文或特殊字符的目录被错误转换，Dataset 可能报 Permission denied。
    因此先写入 ASCII 系统临时路径，关闭文件后复制到同目录唯一 .part 文件，
    最后用 os.replace 原子发布。
    """
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    needs_staging = os.name == "nt" and not str(target).isascii()
    if not needs_staging:
        with netCDF4.Dataset(str(target), "w", format="NETCDF4") as dataset:
            yield dataset
        return

    descriptor, staged_name = tempfile.mkstemp(prefix="bellhop_netcdf_", suffix=".nc")
    os.close(descriptor)
    staged = Path(staged_name)
    staged.unlink()
    part = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        with netCDF4.Dataset(str(staged), "w", format="NETCDF4") as dataset:
            yield dataset
        shutil.copyfile(staged, part)
        os.replace(part, target)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
        try:
            part.unlink()
        except FileNotFoundError:
            pass


# =============================================================================
# 三维 NetCDF 产品导出
# =============================================================================
def write_ray3d_nc(sources: Sequence[Dict[str, Any]], filename: os.PathLike[str] | str) -> int:
    """把多个声源的三维声线写入统一 NetCDF 文件。"""
    all_rays = [(source, ray) for source in sources for ray in source["rays"]]
    if not all_rays:
        raise ValueError("没有可导出的三维声线")
    counts = np.asarray([len(ray["depth_m"]) for _, ray in all_rays], dtype=np.int32)
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    total = int(counts.sum())
    with _netcdf_writer(filename) as ds:
        ds.title = "Multi-bearing BellhopCXX 2D rays rotated into three-dimensional trajectories"
        ds.calculation_method = "independent 2D azimuth sections stitched in source-relative XYZ"
        ds.featureType = "trajectory"
        ds.trajectory_storage = "contiguous ragged array"
        ds.ray_count = len(all_rays)
        ds.point_count = total
        ds.source_count = len(sources)
        ds.createDimension("ray", len(all_rays))
        ds.createDimension("point", total)
        ds.createDimension("rgb", 3)
        _nc_variable(ds, "ray_id", "i4", ("ray",), long_name="one-based ray identifier")[:] = np.arange(1, len(all_rays) + 1)
        _nc_variable(ds, "ray_source_index", "i4", ("ray",), long_name="one-based source index")[:] = [s["index"] for s, _ in all_rays]
        _nc_variable(ds, "ray_bearing_deg", "f4", ("ray",), units="degree", long_name="navigation bearing clockwise from north")[:] = [
            _finite_float(ray.get("bearing_deg"), 0.0) for _, ray in all_rays
        ]
        _nc_variable(ds, "ray_alpha0_deg", "f4", ("ray",), units="degree", long_name="initial elevation angle")[:] = [
            _finite_float(ray.get("alpha0"), 0.0) for _, ray in all_rays
        ]
        _nc_variable(ds, "ray_point_count", "i4", ("ray",))[:] = counts
        _nc_variable(ds, "ray_start_index", "i8", ("ray",), long_name="zero-based point start index")[:] = starts
        _nc_variable(ds, "ray_color_rgb", "f4", ("ray", "rgb"), fill_value=np.nan)[:] = np.asarray([r["color"] for _, r in all_rays])

        point_ray = np.repeat(np.arange(1, len(all_rays) + 1, dtype=np.int32), counts)
        point_source = np.repeat(np.asarray([s["index"] for s, _ in all_rays], dtype=np.int32), counts)
        _nc_variable(ds, "point_ray_id", "i4", ("point",))[:] = point_ray
        _nc_variable(ds, "point_source_index", "i4", ("point",))[:] = point_source

        def cat(name: str, global_xy: bool = False) -> np.ndarray:
            """拼接所有声线中的指定坐标字段。"""
            values = []
            for source, ray in all_rays:
                arr = np.asarray(ray[name])
                if global_xy and name == "x_km":
                    arr = arr + source["x_km"]
                elif global_xy and name == "y_km":
                    arr = arr + source["y_km"]
                values.append(arr)
            return np.concatenate(values)

        _nc_variable(ds, "longitude", "f8", ("point",), units="degrees_east", standard_name="longitude", fill_value=np.nan)[:] = cat("lon")
        _nc_variable(ds, "latitude", "f8", ("point",), units="degrees_north", standard_name="latitude", fill_value=np.nan)[:] = cat("lat")
        _nc_variable(ds, "depth_m", "f4", ("point",), units="m", fill_value=np.nan)[:] = cat("depth_m")
        _nc_variable(ds, "z_m", "f4", ("point",), units="m", fill_value=np.nan)[:] = -cat("depth_m")
        _nc_variable(ds, "local_x_km", "f4", ("point",), units="km", fill_value=np.nan)[:] = cat("x_km", True)
        _nc_variable(ds, "local_y_km", "f4", ("point",), units="km", fill_value=np.nan)[:] = cat("y_km", True)
        _nc_variable(ds, "path_distance_km", "f4", ("point",), units="km", fill_value=np.nan)[:] = cat("path_distance_km")
        _nc_variable(ds, "tl_db", "f4", ("point",), units="dB", fill_value=np.nan)[:] = cat("tl_db")
    return len(all_rays)


def _common_tl_axes(sources: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取多声源传播损失使用的公共方位、深度和距离轴。"""
    if len(sources) != 1:
        raise ValueError("当前多方位 2D 拼接版本只接受一个声源结果")
    first = sources[0]["tl"]
    theta = np.asarray(first["theta_deg"], dtype=np.float64)
    ranges = np.asarray(first["range_km"], dtype=np.float64)
    depth = np.asarray(first["depth_m"], dtype=np.float64)
    return theta, depth, ranges


def _tl_layers_on_depth(source: Dict[str, Any], target_depth: np.ndarray) -> np.ndarray:
    """把单个声源的 depth/theta/range TL 插值到统一深度轴上。"""
    source_depth = np.asarray(source["tl"]["depth_m"], dtype=np.float64)
    layers = np.asarray(source["tl"]["layers"], dtype=np.float64)
    if source_depth.shape == target_depth.shape and np.allclose(source_depth, target_depth, rtol=0.0, atol=1e-5):
        return layers
    flat = layers.reshape(len(source_depth), -1)
    out = np.full((len(target_depth), flat.shape[1]), np.nan, dtype=np.float64)
    for column in range(flat.shape[1]):
        values = flat[:, column]
        good = np.isfinite(values) & np.isfinite(source_depth)
        if np.count_nonzero(good) >= 2:
            z = source_depth[good]
            order = np.argsort(z)
            out[:, column] = np.interp(target_depth, z[order], values[good][order], left=np.nan, right=np.nan)
    return out.reshape(len(target_depth), *layers.shape[1:])


def write_tl3d_nc(sources: Sequence[Dict[str, Any]], filename: os.PathLike[str] | str) -> None:
    """把多声源极坐标三维传播损失写入 NetCDF 文件。"""
    theta, depth, ranges = _common_tl_axes(sources)
    data = np.stack([_tl_layers_on_depth(source, depth) for source in sources], axis=0)
    bottom = np.stack([np.asarray(s["tl"]["bottom_depth_m"]) for s in sources], axis=0)
    with _netcdf_writer(filename) as ds:
        ds.title = "Stitched multi-bearing BellhopCXX 2D full-depth polar transmission loss"
        ds.calculation_method = "tl_db(depth,theta,range) assembled from independent 2D sections"
        ds.coordinate_rule = "tl_db(source,depth,theta,range)"
        ds.createDimension("source", len(sources))
        ds.createDimension("depth", len(depth))
        ds.createDimension("theta", len(theta))
        ds.createDimension("range", len(ranges))
        _nc_variable(ds, "source_index", "i4", ("source",))[:] = [s["index"] for s in sources]
        _nc_variable(ds, "source_longitude", "f8", ("source",), units="degrees_east")[:] = [s["lon"] for s in sources]
        _nc_variable(ds, "source_latitude", "f8", ("source",), units="degrees_north")[:] = [s["lat"] for s in sources]
        _nc_variable(ds, "source_x_km", "f4", ("source",), units="km")[:] = [s["x_km"] for s in sources]
        _nc_variable(ds, "source_y_km", "f4", ("source",), units="km")[:] = [s["y_km"] for s in sources]
        _nc_variable(ds, "depth_m", "f4", ("depth",), units="m")[:] = depth
        _nc_variable(ds, "theta_deg", "f4", ("theta",), units="degree")[:] = theta
        _nc_variable(ds, "range_km", "f4", ("range",), units="km")[:] = ranges
        _nc_variable(ds, "bottom_depth_m", "f4", ("source", "theta", "range"), units="m", fill_value=np.nan)[:] = bottom
        _nc_variable(ds, "tl_db", "f4", ("source", "depth", "theta", "range"), units="dB", fill_value=np.nan)[:] = data

def _source_lookup_on_master_grid(
    source: Dict[str, Any],
    master: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在公共笛卡尔网格上预先计算相对声源的 theta/range 索引。"""
    xs = np.asarray(master["x_km"], dtype=np.float64)
    ys = np.asarray(master["y_km"], dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    tl = source["tl"]
    theta = np.asarray(tl["theta_deg"], dtype=np.float64)
    ranges = np.asarray(tl["range_km"], dtype=np.float64)
    dx = xx - _finite_float(source.get("x_km"), 0.0)
    dy = yy - _finite_float(source.get("y_km"), 0.0)
    rr = np.hypot(dx, dy)
    bearing = (np.rad2deg(np.arctan2(dx, dy)) + 360.0) % 360.0
    it = np.abs(((theta[:, None, None] - bearing[None, :, :] + 180.0) % 360.0) - 180.0).argmin(axis=0)
    ir = _nearest_sorted_indices(ranges, rr)
    radius = _finite_float(source.get("calculation_radius_km"), float(ranges[-1]))
    valid = rr <= radius + 1.0e-7
    return it, ir, valid


def write_tl_cartesian_nc(
    sources: Sequence[Dict[str, Any]],
    master: Dict[str, Any],
    params: Dict[str, Any],
    filename: os.PathLike[str] | str,
) -> None:
    """把拼接后的单声源 TL 三维场写到笛卡尔网格上。"""
    _, depth, _ = _common_tl_axes(sources)
    xs = np.asarray(master["x_km"], dtype=np.float64)
    ys = np.asarray(master["y_km"], dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    bottom = np.asarray(master["bottom_depth_m"], dtype=np.float64)
    calculation_mask = np.asarray(master.get("calculation_mask", np.ones_like(bottom)), dtype=bool)
    source = sources[0]
    lookup = _source_lookup_on_master_grid(source, master)
    layers = [_tl_layers_on_depth(source, depth) for source in sources]
    lon, lat = local_offset_to_lonlat(xx, yy, sources[0]["lon"], sources[0]["lat"])

    with _netcdf_writer(filename) as ds:
        ds.title = "Multi-bearing Bellhop 2D stitched full-depth transmission loss"
        ds.createDimension("source", len(sources))
        ds.createDimension("depth", len(depth))
        ds.createDimension("y", len(ys))
        ds.createDimension("x", len(xs))
        _nc_variable(ds, "source_index", "i4", ("source",))[:] = [s["index"] for s in sources]
        _nc_variable(ds, "source_longitude", "f8", ("source",), units="degrees_east")[:] = [s["lon"] for s in sources]
        _nc_variable(ds, "source_latitude", "f8", ("source",), units="degrees_north")[:] = [s["lat"] for s in sources]
        _nc_variable(ds, "source_x_km", "f4", ("source",), units="km")[:] = [s["x_km"] for s in sources]
        _nc_variable(ds, "source_y_km", "f4", ("source",), units="km")[:] = [s["y_km"] for s in sources]
        _nc_variable(ds, "depth_m", "f4", ("depth",), units="m")[:] = depth
        _nc_variable(ds, "x_km", "f4", ("x",), units="km")[:] = xs
        _nc_variable(ds, "y_km", "f4", ("y",), units="km")[:] = ys
        _nc_variable(ds, "longitude", "f8", ("y", "x"), units="degrees_east", fill_value=np.nan)[:] = lon
        _nc_variable(ds, "latitude", "f8", ("y", "x"), units="degrees_north", fill_value=np.nan)[:] = lat
        _nc_variable(ds, "bottom_depth_m", "f4", ("y", "x"), units="m", fill_value=np.nan)[:] = bottom
        _nc_variable(ds, "calculation_mask", "i1", ("y", "x"))[:] = calculation_mask.astype(np.int8)
        tl_var = _nc_variable(ds, "tl_db", "f4", ("depth", "y", "x"), units="dB", fill_value=np.nan)

        for iz, depth_value in enumerate(depth):
            it, ir, geometric_valid = lookup
            values = np.asarray(layers[0][iz], dtype=np.float64)[it, ir]
            valid = geometric_valid & calculation_mask & np.isfinite(values) & (float(depth_value) <= bottom)
            field = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
            field[valid] = values[valid]
            tl_var[iz, :, :] = field.astype(np.float32)



# =============================================================================
# 二维剖面和水平场 NetCDF 导出
# =============================================================================
def _sample_horizontal_tl(
    source: Dict[str, Any],
    master: Dict[str, Any],
    depth_m: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """在指定接收深度上采样单声源拼接后的 TL 水平场。"""
    xs = np.asarray(master["x_km"], dtype=np.float64)
    ys = np.asarray(master["y_km"], dtype=np.float64)
    field = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    calculation_mask = np.asarray(master.get("calculation_mask", np.ones_like(field)), dtype=bool)
    bottom = np.asarray(master["bottom_depth_m"], dtype=np.float64)
    requested_depth = max(0.0, float(depth_m))
    tl = source["tl"]
    depths = np.asarray(tl["depth_m"], dtype=np.float64)
    layers = np.asarray(tl["layers"], dtype=np.float64)
    if depths.size == 0 or layers.shape[0] != depths.size:
        return field, bottom, requested_depth
    order = np.argsort(depths)
    depths = depths[order]
    layers = layers[order]
    if requested_depth < float(depths[0]) - 1.0e-7 or requested_depth > float(depths[-1]) + 1.0e-7:
        return field, bottom, requested_depth
    hi = int(np.searchsorted(depths, requested_depth, side="left"))
    if hi <= 0:
        sampled_layer = layers[0]
    elif hi >= len(depths):
        sampled_layer = layers[-1]
    elif abs(float(depths[hi]) - requested_depth) <= 1.0e-7:
        sampled_layer = layers[hi]
    else:
        lo = hi - 1
        z0, z1 = float(depths[lo]), float(depths[hi])
        weight = 0.0 if z1 <= z0 else (requested_depth - z0) / (z1 - z0)
        a, b = layers[lo], layers[hi]
        sampled_layer = np.full_like(a, np.nan, dtype=np.float64)
        finite_pair = np.isfinite(a) & np.isfinite(b)
        sampled_layer[finite_pair] = a[finite_pair] + (b[finite_pair] - a[finite_pair]) * weight

    it, ir, geometric_valid = _source_lookup_on_master_grid(source, master)
    values = sampled_layer[it, ir]
    valid = geometric_valid & calculation_mask & np.isfinite(values) & (requested_depth <= bottom)
    field[valid] = values[valid]
    return field, bottom, requested_depth


def write_tl_horizontal_nc(
    source: Dict[str, Any],
    master: Dict[str, Any],
    params: Dict[str, Any],
    filename: os.PathLike[str] | str,
) -> Tuple[np.ndarray, float]:
    """在指定深度采样并导出水平二维传播损失 NetCDF。"""
    requested = _finite_float(params.get("horizontal_tl_depth_m"), params.get("source_depth_m", 150.0))
    horizontal, bottom, actual = _sample_horizontal_tl(source, master, requested)
    xs, ys = np.asarray(master["x_km"]), np.asarray(master["y_km"])
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = local_offset_to_lonlat(xx, yy, source["lon"], source["lat"])
    with _netcdf_writer(filename) as ds:
        ds.title = "Multi-bearing Bellhop 2D stitched horizontal transmission loss"
        ds.source_longitude = float(source["lon"])
        ds.source_latitude = float(source["lat"])
        ds.createDimension("x", len(xs))
        ds.createDimension("y", len(ys))
        _nc_variable(ds, "x_km", "f4", ("x",), units="km")[:] = xs
        _nc_variable(ds, "y_km", "f4", ("y",), units="km")[:] = ys
        _nc_variable(ds, "longitude", "f8", ("y", "x"), units="degrees_east", fill_value=np.nan)[:] = lon
        _nc_variable(ds, "latitude", "f8", ("y", "x"), units="degrees_north", fill_value=np.nan)[:] = lat
        depth_var = ds.createVariable("depth_m", "f4")
        depth_var.assignValue(actual)
        depth_var.units = "m"
        _nc_variable(ds, "bottom_depth_m", "f4", ("y", "x"), units="m", fill_value=np.nan)[:] = bottom
        tl_var = _nc_variable(ds, "tl_db", "f4", ("y", "x"), units="dB", fill_value=np.nan)
        tl_var.long_name = "stitched single-source transmission loss"
        tl_var[:] = horizontal
    return horizontal, actual


def write_tl_section_nc(
    source: Dict[str, Any], params: Dict[str, Any], filename: os.PathLike[str] | str,
) -> None:
    """在指定深度采样并导出水平二维传播损失 NetCDF。"""
    """写出用户选择的单声源垂直剖面。"""
    tl = source["tl"]
    depth = np.asarray(tl["depth_m"], dtype=np.float64)
    ranges = np.asarray(tl["range_km"], dtype=np.float64)
    wanted = _finite_float(params.get("selected_section_theta_deg"), params.get("beta_min", 0.0)) % 360.0
    theta = np.asarray(tl["theta_deg"], dtype=np.float64)
    section_index = int(np.abs(((theta - wanted + 180.0) % 360.0) - 180.0).argmin())
    used_theta = float(theta[section_index])
    section = np.asarray(tl["layers"], dtype=np.float64)[:, section_index, :].T
    section_bottom = np.asarray(tl["bottom_depth_m"], dtype=np.float64)[section_index, :]
    angle = math.radians(used_theta)
    section_x = ranges * math.sin(angle)
    section_y = ranges * math.cos(angle)

    with _netcdf_writer(filename) as ds:
        ds.title = "Selected vertical section from stitched Bellhop 2D TL"
        ds.source_depth_m = float(source.get("source_depth_m", params.get("source_depth_m", 100.0)))
        ds.tl_field_type = str(params.get("tl_run_type", "I"))
        ds.frequency_hz = float(params.get("freq_hz", 5000.0))
        if "ssp_depth_m" in source and "ssp_sound_speed_mps" in source:
            ds.createDimension("ssp_depth", len(source["ssp_depth_m"]))
            _nc_variable(ds, "ssp_depth_m", "f8", ("ssp_depth",), units="m")[:] = source["ssp_depth_m"]
            _nc_variable(ds, "ssp_sound_speed_mps", "f8", ("ssp_depth",), units="m s-1")[:] = source["ssp_sound_speed_mps"]
            ds.travel_time_profile_note = "source-local SSP; not the full range-dependent SSP"
        boundaries = source.get("reverberation_boundaries", [])
        if boundaries:
            native = min(boundaries, key=lambda item: abs(((item["bearing_deg"]-used_theta+180.) % 360.)-180.))
            ds.createDimension("boundary_range", len(native["range_km"]))
            _nc_variable(ds, "boundary_range_m", "f8", ("boundary_range",), units="m")[:] = np.asarray(native["range_km"])*1000.
            _nc_variable(ds, "boundary_bottom_m", "f8", ("boundary_range",), units="m")[:] = native["bottom_depth_m"]
            _nc_variable(ds, "boundary_surface_m", "f8", ("boundary_range",), units="m")[:] = native["surface_depth_m"]
            if "ssp_depth" in ds.dimensions and "sound_speed_mps" in native:
                _nc_variable(ds, "section_sound_speed_mps", "f8", ("ssp_depth", "boundary_range"), units="m s-1")[:] = native["sound_speed_mps"]
                ds.travel_time_profile_note = "range-dependent section SSP saved with native boundary ranges"
            ds.boundary_geometry_source = "Bellhop environment native boundary polyline"
        ds.section_reference = "section starts at the source and follows selected theta"
        ds.source_longitude = float(source["lon"])
        ds.source_latitude = float(source["lat"])
        ds.createDimension("range", len(ranges))
        ds.createDimension("depth", len(depth))
        angle_var = ds.createVariable("section_theta_deg", "f4")
        angle_var.assignValue(wanted)
        angle_var.units = "degree"
        solved_angle_var = ds.createVariable("solved_theta_deg", "f4")
        solved_angle_var.assignValue(used_theta)
        solved_angle_var.units = "degree"
        _nc_variable(ds, "range_km", "f4", ("range",), units="km")[:] = ranges
        _nc_variable(ds, "section_x_km", "f4", ("range",), units="km")[:] = section_x
        _nc_variable(ds, "section_y_km", "f4", ("range",), units="km")[:] = section_y
        lon, lat = local_offset_to_lonlat(section_x, section_y, source["lon"], source["lat"])
        _nc_variable(ds, "longitude", "f8", ("range",), units="degrees_east", fill_value=np.nan)[:] = lon
        _nc_variable(ds, "latitude", "f8", ("range",), units="degrees_north", fill_value=np.nan)[:] = lat
        _nc_variable(ds, "depth_m", "f4", ("depth",), units="m")[:] = depth
        _nc_variable(ds, "bottom_depth_m", "f4", ("range",), units="m", fill_value=np.nan)[:] = section_bottom
        _nc_variable(ds, "tl_db", "f4", ("range", "depth"), units="dB", fill_value=np.nan)[:] = section

def _section_ray_segments(source: Dict[str, Any], wanted: float) -> List[Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
    """提取最接近指定方位的声线剖面片段。"""
    theta_grid = np.asarray(source["tl"]["theta_deg"])
    tolerance = max(3.0, min(10.0, 180.0 / max(18, len(theta_grid))))
    angle = math.radians(wanted)
    ux, uy = math.sin(angle), math.cos(angle)
    selected = []
    for ray in source["rays"]:
        x, y, depth = np.asarray(ray["x_km"]), np.asarray(ray["y_km"]), np.asarray(ray["depth_m"])
        far = int(np.argmax(np.hypot(x, y)))
        bearing = (math.degrees(math.atan2(x[far], y[far])) + 360.0) % 360.0
        diff = abs(((bearing - wanted + 180.0) % 360.0) - 180.0)
        if diff > tolerance:
            continue
        along = x * ux + y * uy
        good = (along >= -1e-6) & np.isfinite(along) & np.isfinite(depth)
        if np.count_nonzero(good) >= 2:
            selected.append((ray, along[good], depth[good]))
    return selected


def write_ray_section_nc(
    sources: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    filename: os.PathLike[str] | str,
) -> None:
    """把选定方位的二维声线剖面写入 NetCDF 文件。"""
    wanted = _finite_float(params.get("selected_section_theta_deg"), params.get("beta_min", 0.0)) % 360.0
    segments: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]] = []
    for source in sources:
        for ray, ranges, depths in _section_ray_segments(source, wanted):
            segments.append((source, ray, ranges, depths))
    counts = np.asarray([len(x[2]) for x in segments], dtype=np.int32)
    if not len(counts):
        counts = np.asarray([1], dtype=np.int32)
        segments = [(sources[0], {"id": 0}, np.asarray([np.nan]), np.asarray([np.nan]))]
        actual = 0
    else:
        actual = len(counts)
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    with _netcdf_writer(filename) as ds:
        ds.title = "BellhopCXX selected-section rays"
        ds.section_theta_deg = wanted
        ds.actual_ray_count = actual
        ds.createDimension("ray", len(segments))
        ds.createDimension("point", int(counts.sum()))
        _nc_variable(ds, "ray_source_index", "i4", ("ray",))[:] = [s["index"] for s, _, _, _ in segments]
        _nc_variable(ds, "ray_id", "i4", ("ray",))[:] = [r.get("id", 0) for _, r, _, _ in segments]
        _nc_variable(ds, "ray_point_count", "i4", ("ray",))[:] = counts
        _nc_variable(ds, "ray_start_index", "i8", ("ray",))[:] = starts
        _nc_variable(ds, "point_ray_index", "i4", ("point",))[:] = np.repeat(np.arange(1, len(segments) + 1), counts)
        _nc_variable(ds, "range_km", "f4", ("point",), units="km", fill_value=np.nan)[:] = np.concatenate([x[2] for x in segments])
        _nc_variable(ds, "depth_m", "f4", ("point",), units="m", fill_value=np.nan)[:] = np.concatenate([x[3] for x in segments])


# =============================================================================
# 前端 JSON 缓存
# =============================================================================
def _decimate_indices(count: int, maximum: int) -> np.ndarray:
    """生成不超过指定数量且覆盖首尾位置的抽稀索引。"""
    if count <= maximum:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, maximum).round().astype(int))


def build_scene_json(
    sources: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    master: Dict[str, Any],
    horizontal_tl: np.ndarray,
    actual_depth: float,
) -> Dict[str, Any]:
    """构造三维场景使用的轻量级 JSON 数据。"""
    selection = []
    for source in sources:
        selection.append({
            "index": source["index"], "x_km": source["x_km"], "y_km": source["y_km"],
            "is_source": True, "lat": source["lat"], "lon": source["lon"],
            "color": SOURCE_COLORS[(source["index"] - 1) % len(SOURCE_COLORS)],
        })
    scene_rays: List[Dict[str, Any]] = []
    for source in sources:
        ray_indices = _decimate_indices(len(source["rays"]), 320)
        for ir in ray_indices:
            ray = source["rays"][int(ir)]
            idx = _decimate_indices(len(ray["depth_m"]), 420)
            points = np.column_stack([
                np.asarray(ray["x_km"])[idx] + source["x_km"],
                np.asarray(ray["y_km"])[idx] + source["y_km"],
                np.asarray(ray["depth_m"])[idx],
            ])
            scene_rays.append({
                "points": _json_array(points, 5),
                "color": ray["color"],
                "source_index": source["index"],
                "bearing_deg": _finite_float(ray.get("bearing_deg"), 0.0),
                "alpha0_deg": _finite_float(ray.get("alpha0"), 0.0),
                "ntop": ray["ntop"],
                "nbot": ray["nbot"],
            })

    xs = np.asarray(master["x_km"])
    ys = np.asarray(master["y_km"])
    xx, yy = np.meshgrid(xs, ys)
    bottom = np.asarray(master["bottom_depth_m"])
    terrain_mask = np.asarray(master.get("terrain_mask", np.ones_like(bottom)), dtype=np.int8)
    thermo = np.asarray(master.get("thermocline_depth_m", np.full_like(bottom, np.nan)))
    thermo_gradient = np.asarray(
        master.get("thermocline_gradient_c_per_m", np.full_like(bottom, np.nan)),
        dtype=np.float64,
    )
    calculation_mask = np.asarray(master.get("calculation_mask", np.ones_like(bottom)), dtype=np.int8)
    finite_bottom = bottom[np.isfinite(bottom)]
    bounds = {
        "x": [float(xs[0]), float(xs[-1])], "y": [float(ys[0]), float(ys[-1])],
        "depth": [0.0, float(np.max(finite_bottom)) if finite_bottom.size else 5000.0],
    }
    finite_tl = horizontal_tl[np.isfinite(horizontal_tl) & (calculation_mask > 0)]
    if finite_tl.size:
        lo, hi = np.nanpercentile(finite_tl, [15, 85])
        levels = np.linspace(lo, hi, 6)
    else:
        levels = np.asarray([])

    finite_gradient = thermo_gradient[
        np.isfinite(thermo_gradient) & np.isfinite(thermo) & (calculation_mask > 0)
    ]
    if finite_gradient.size:
        grad_lo, grad_hi = np.nanpercentile(finite_gradient, [10, 90])
        if grad_hi > grad_lo + 1.0e-9:
            gradient_levels = np.linspace(grad_lo, grad_hi, 7)
        else:
            gradient_levels = np.asarray([float(grad_lo)])
    else:
        gradient_levels = np.asarray([])
    return {
        "version": "scene3d-local-km-python-v31-multibearing-2d-stitched-field",
        "coordinate_mode": "local_km",
        "source_mode": "single_source",
        "calculation_method": "multi_bearing_bellhop_2d_stitched_3d",
        "azimuths_deg": _json_array(sources[0]["tl"]["theta_deg"], 5),
        "units": {"horizontal": "km", "depth": "m"},
        "source_lon": sources[0]["lon"], "source_lat": sources[0]["lat"],
        "source_depth_m": _finite_float(params.get("source_depth_m"), 150.0),
        "source_x_km": 0.0, "source_y_km": 0.0,
        "radius_km": _finite_float(params.get("radius_km"), 50.0),
        "selection_point_count": len(selection),
        "calculation_range_km": max(_finite_float(s.get("calculation_radius_km")) for s in sources),
        "max_water_depth_m": bounds["depth"][1],
        "selection_points": selection,
        "rays": scene_rays,
        "bathymetry": {
            "has_data": True, "x_km": _json_array(xx, 5), "y_km": _json_array(yy, 5),
            "depths": _json_array(bottom, 3), "terrain_mask": terrain_mask.tolist(),
            "mask_padding_km": 0.0,
            "depth_min_m": float(np.nanmin(bottom)), "depth_max_m": float(np.nanmax(bottom)),
        },
        "thermocline": {
            "has_data": bool(np.any(np.isfinite(thermo))), "mode": "maximum_abs_temperature_gradient",
            # 同时保留二维坐标网格，便于旧版前端读取；新版前端也支持 1D/2D 坐标。
            "x_km": _json_array(xx, 5), "y_km": _json_array(yy, 5),
            "depths": _json_array(thermo, 3),
            "gradient_c_per_m": _json_array(thermo_gradient, 6),
            "gradient_levels_c_per_m": _json_array(gradient_levels, 6),
            "calculation_mask": calculation_mask.tolist(), "contours": [],
            "depth_min_m": float(np.nanmin(thermo)) if np.any(np.isfinite(thermo)) else None,
            "depth_max_m": float(np.nanmax(thermo)) if np.any(np.isfinite(thermo)) else None,
        },
        "tl_contours": {
            "depth_m": actual_depth, "contours": [],
            # 传播损失 TL 回退等值线使用 1D x/y 轴，避免把二维 meshgrid 误转成 NaN。
            "grid_x_km": _json_array(xs, 5), "grid_y_km": _json_array(ys, 5),
            "grid_tl_db": _json_array(horizontal_tl, 3),
            "calculation_mask": calculation_mask.tolist(),
            "levels_db": _json_array(levels, 3),
        },
        "bounds": bounds,
    }


def build_interactive_json(
    sources: Sequence[Dict[str, Any]], params: Dict[str, Any], master: Dict[str, Any]
) -> Dict[str, Any]:
    """构造前端交互面板使用的声线、声场和剖面数据。"""
    source_items = []
    for source in sources:
        ray_items = []
        for ir in _decimate_indices(len(source["rays"]), 240):
            ray = source["rays"][int(ir)]
            idx = _decimate_indices(len(ray["depth_m"]), 360)
            ray_items.append({
                "x_km": _json_array(np.asarray(ray["x_km"])[idx], 5),
                "y_km": _json_array(np.asarray(ray["y_km"])[idx], 5),
                "depth_m": _json_array(np.asarray(ray["depth_m"])[idx], 3),
                "color": ray["color"],
                "reflection_type": ray.get("reflection_type", "none"),
                "bearing_deg": _finite_float(ray.get("bearing_deg"), 0.0),
                "alpha0_deg": _finite_float(ray.get("alpha0"), 0.0),
                "ray_id": int(ray.get("id", 0)),
                "ntop": int(ray.get("ntop", 0)),
                "nbot": int(ray.get("nbot", 0)),
            })
        tl = source["tl"]
        source_items.append({
            "index": source["index"], "lat": source["lat"], "lon": source["lon"],
            "x_km": source["x_km"], "y_km": source["y_km"],
            "source_depth_m": _finite_float(params.get("source_depth_m"), 150.0),
            "source_bottom_depth_m": _finite_float(source.get("source_bottom_depth_m"), np.nan),
            "ssp": {
                "depth_m": _json_array(source["ssp_depth_m"], 3),
                "sound_speed_mps": _json_array(source["ssp_sound_speed_mps"], 4),
            },
            "tl": {
                "theta_deg": _json_array(tl["theta_deg"], 5),
                "depth_m": _json_array(tl["depth_m"], 3),
                "range_km": _json_array(tl["range_km"], 5),
                "layers": _json_array(tl["layers"], 4),
                "bottom_depth_m": _json_array(tl["bottom_depth_m"], 3),
            },
            "rays": ray_items,
        })
    bottom = np.asarray(master["bottom_depth_m"])
    return {
        "version": "interactive-panels-python-v31-multibearing-2d-stitched-field",
        "source_mode": "single_source",
        "calculation_method": "multi_bearing_bellhop_2d_stitched_3d",
        "azimuths_deg": _json_array(sources[0]["tl"]["theta_deg"], 5),
        "radius_km": _finite_float(params.get("radius_km"), 50.0),
        "default_depth_m": _finite_float(params.get("horizontal_tl_depth_m"), params.get("source_depth_m", 150.0)),
        "default_theta_deg": _finite_float(params.get("selected_section_theta_deg"), params.get("beta_min", 0.0)),
        "tl_caxis": [
            _finite_float(params.get("tl_caxis_min_db"), 50.0),
            _finite_float(params.get("tl_caxis_max_db"), 100.0),
        ],
        "sources": source_items,
        "warnings": [],
        "master": {
            "x_km": _json_array(master["x_km"], 5), "y_km": _json_array(master["y_km"], 5),
            "bottom_depth_m": _json_array(bottom, 3),
            "terrain_mask": np.asarray(master.get("terrain_mask", np.ones_like(bottom)), dtype=np.int8).tolist(),
            "calculation_mask": np.asarray(master.get("calculation_mask", np.ones_like(bottom)), dtype=np.int8).tolist(),
        },
        "calculation_note": "Bellhop 2D 方位剖面旋转并拼接得到单声源三维传播损失。",
    }



# =============================================================================
# 导出总入口
# =============================================================================
def export_job_results(
    sources: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    job_dir: os.PathLike[str] | str,
    master: Dict[str, Any],
) -> Dict[str, Any]:
    """写出公开 NetCDF 产品和浏览器端 JSON 缓存。"""
    if len(sources) != 1:
        raise ValueError("当前多方位 2D 拼接版本必须且只能导出一个声源结果")
    out = Path(job_dir)
    source = sources[0]
    for source in sources:
        attach_tl_to_rays(source)

    ray_count = write_ray3d_nc(sources, out / "ray3d_wgs84.nc")
    write_ray_section_nc(sources, params, out / "ray2d_section.nc")
    write_tl3d_nc(sources, out / "tl3d_depth_theta_range.nc")
    write_tl_section_nc(source, params, out / "tl2d_section.nc")

    # 三维 TL / 声线响应已由接口写回 response_<Plugin>.json。
    # 不再生成当前后端未消费的笛卡尔 TL、水平 TL 和 scene3d.json。
    interactive = build_interactive_json(sources, params, master)
    _write_json(out / "interactive_panels.json", interactive)
    return {
        "ray_count": ray_count,
        "source_count": len(sources),
        "files": [
            "ray3d_wgs84.nc", "ray2d_section.nc", "tl3d_depth_theta_range.nc",
            "tl2d_section.nc", "interactive_panels.json",
        ],
    }
