# -*- coding: utf-8 -*-
"""解析 RAM 的 `tl.line`、Fortran 二进制 `tl.grid` 并导出 NetCDF。"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

import netCDF4
import numpy as np


_HEADER_FORMAT = "5fi2fi2f2ifi"
_HEADER_NAMES = (
    "frequency_hz", "source_depth_m", "receiver_depth_m", "max_range_m", "range_step_m",
    "range_output_decimation", "max_depth_m", "depth_step_m", "depth_output_decimation",
    "output_depth_max_m", "reference_sound_speed_mps", "pade_terms", "stability_constraints",
    "stability_disable_range_m", "output_depth_count",
)


def _retry_windows_file_operation(operation, attempts: int = 20):
    """等待 Windows 延迟释放 NetCDF 临时文件，避免 WinError 32。"""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == attempts - 1:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def _cleanup_temporary_file(path: Path) -> None:
    try:
        _retry_windows_file_operation(lambda: path.unlink(missing_ok=True))
    except OSError:
        # 清理失败不应覆盖已经成功生成的声场结果。
        pass


def _read_record(stream, endian: str) -> bytes | None:
    """读取一个带四字节前后记录标记的 Fortran 顺序记录。"""
    marker = stream.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise ValueError("tl.grid 的 Fortran 记录头不完整。")
    length = struct.unpack(f"{endian}i", marker)[0]
    if length <= 0 or length > 100_000_000:
        raise ValueError("tl.grid 的 Fortran 记录长度无效。")
    payload = stream.read(length)
    trailer = stream.read(4)
    if len(payload) != length or len(trailer) != 4 or struct.unpack(f"{endian}i", trailer)[0] != length:
        raise ValueError("tl.grid 的 Fortran 记录前后标记不一致。")
    return payload


def _detect_endian(path: Path) -> str:
    """根据首记录长度识别 RAM 二进制文件的字节序。"""
    raw = path.read_bytes()[:4]
    if len(raw) != 4:
        raise ValueError("tl.grid 文件为空。")
    expected = struct.calcsize(f"<{_HEADER_FORMAT}")
    if struct.unpack("<i", raw)[0] == expected:
        return "<"
    if struct.unpack(">i", raw)[0] == expected:
        return ">"
    raise ValueError("tl.grid 首记录与当前 RAM 1.5 格式不匹配。")


def read_tl_grid(path: str | Path) -> Dict[str, Any]:
    """读取 RAM 1.5 的 `tl.grid`，重建距离轴、深度轴和二维 TL。"""
    source = Path(path)
    endian = _detect_endian(source)
    with source.open("rb") as stream:
        header_record = _read_record(stream, endian)
        header_values = struct.unpack(f"{endian}{_HEADER_FORMAT}", header_record)
        header = dict(zip(_HEADER_NAMES, header_values))
        depth_count = int(header["output_depth_count"])
        rows = []
        while True:
            record = _read_record(stream, endian)
            if record is None:
                break
            if len(record) != depth_count * 4:
                raise ValueError("tl.grid 的 TL 记录长度与输出深度数不一致。")
            rows.append(np.frombuffer(record, dtype=f"{endian}f4").astype(np.float64))
    if not rows:
        raise ValueError("tl.grid 中没有传播损失记录。")
    tl_by_range = np.vstack(rows)
    range_spacing = float(header["range_step_m"]) * int(header["range_output_decimation"])
    depth_spacing = float(header["depth_step_m"]) * int(header["depth_output_decimation"])
    return {
        "header": header,
        "range_m": range_spacing * np.arange(1, len(rows) + 1, dtype=np.float64),
        "depth_m": depth_spacing * np.arange(1, depth_count + 1, dtype=np.float64),
        "tl_db": tl_by_range.T,
        "endian": "little" if endian == "<" else "big",
    }


def read_tl_line(path: str | Path) -> Dict[str, np.ndarray]:
    """读取接收深度处的 `tl.line` 文本结果。"""
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] < 2 or values.shape[0] == 0:
        raise ValueError("tl.line 未包含有效的距离和 TL 数据。")
    return {"range_m": values[:, 0], "tl_db": values[:, 1]}


@contextmanager
def _netcdf_writer(path: str | Path):
    """通过 ASCII 临时路径兼容 Windows 中文目录下的 NetCDF 写入。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" or str(target).isascii():
        with netCDF4.Dataset(str(target), "w", format="NETCDF4") as dataset:
            yield dataset
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix="ram_netcdf_", suffix=".nc")
    os.close(descriptor)
    temporary = Path(temporary_name)
    _retry_windows_file_operation(lambda: temporary.unlink(missing_ok=True))
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        with netCDF4.Dataset(str(temporary), "w", format="NETCDF4") as dataset:
            yield dataset
        _retry_windows_file_operation(lambda: shutil.copyfile(temporary, partial))
        _retry_windows_file_operation(lambda: os.replace(partial, target))
    finally:
        _cleanup_temporary_file(temporary)
        _cleanup_temporary_file(partial)


def write_tl_netcdf(grid: Mapping[str, Any], line: Mapping[str, Any], path: str | Path) -> Path:
    """把 RAM 二维 TL 网格和指定接收深度曲线写入统一 NetCDF。"""
    target = Path(path)
    ranges = np.asarray(grid["range_m"], dtype=np.float64)
    depths = np.asarray(grid["depth_m"], dtype=np.float64)
    tl = np.asarray(grid["tl_db"], dtype=np.float64)
    with _netcdf_writer(target) as dataset:
        dataset.title = "RAM parabolic-equation transmission loss"
        dataset.calculation_model = "Range-dependent Acoustic Model"
        for key, value in grid["header"].items():
            setattr(dataset, key, value)
        dataset.createDimension("depth", len(depths))
        dataset.createDimension("range", len(ranges))
        dataset.createDimension("line_range", len(line["range_m"]))
        depth_var = dataset.createVariable("depth_m", "f4", ("depth",))
        range_var = dataset.createVariable("range_m", "f4", ("range",))
        tl_var = dataset.createVariable("tl_db", "f4", ("depth", "range"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        line_range = dataset.createVariable("line_range_m", "f4", ("line_range",))
        line_tl = dataset.createVariable("line_tl_db", "f4", ("line_range",), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        depth_var.units = "m"
        range_var.units = "m"
        tl_var.units = "dB"
        line_range.units = "m"
        line_tl.units = "dB"
        depth_var[:] = depths.astype(np.float32)
        range_var[:] = ranges.astype(np.float32)
        tl_var[:] = tl.astype(np.float32)
        line_range[:] = np.asarray(line["range_m"], dtype=np.float32)
        line_tl[:] = np.asarray(line["tl_db"], dtype=np.float32)
    return target


def result_statistics(grid: Mapping[str, Any], line: Mapping[str, Any]) -> Dict[str, Any]:
    """生成 RAM 结果的尺寸、范围和有限 TL 统计。"""
    tl = np.asarray(grid["tl_db"], dtype=np.float64)
    finite = tl[np.isfinite(tl)]
    return {
        "frequency_hz": float(grid["header"]["frequency_hz"]),
        "range_count": int(tl.shape[1]),
        "depth_count": int(tl.shape[0]),
        "range_min_m": float(np.min(grid["range_m"])),
        "range_max_m": float(np.max(grid["range_m"])),
        "depth_min_m": float(np.min(grid["depth_m"])),
        "depth_max_m": float(np.max(grid["depth_m"])),
        "tl_min_db": float(np.min(finite)) if finite.size else math.nan,
        "tl_max_db": float(np.max(finite)) if finite.size else math.nan,
        "line_count": int(len(line["range_m"])),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """以 UTF-8 和中文友好格式写出 JSON 摘要。"""
    target = Path(path)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
