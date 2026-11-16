# -*- coding: utf-8 -*-
"""读取 Kraken/FIELD 结果并转换为便于后端使用的文件。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

import netCDF4
import numpy as np

from backworker.bellhop_model.results import read_shd_bin


_MODE_LINE = re.compile(
    r"^\s*(\d+)\s+([+-]?[\d.]+(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?[\d.]+(?:[Ee][+-]?\d+)?)\s+([+-]?[\d.]+(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?[\d.]+(?:[Ee][+-]?\d+)?)\s*$"
)


def parse_mode_summary(print_file: str | Path) -> Dict[str, Any]:
    """从 Kraken 打印文件中提取模态数、水平波数和相速度摘要。"""
    path = Path(print_file)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _MODE_LINE.match(line)
            if match:
                rows.append({
                    "mode": int(match.group(1)),
                    "wavenumber_real": float(match.group(2)),
                    "wavenumber_imag": float(match.group(3)),
                    "phase_speed_mps": float(match.group(4)),
                    "group_speed_mps": float(match.group(5)),
                })
    return {
        "mode_count": len(rows),
        "modes": rows,
    }


def _read_legacy_field_shd(shade_file: str | Path) -> Dict[str, Any]:
    """兼容 2018 FIELD 未写 PlotType 标记的旧版矩形 SHD 文件。"""
    source = Path(shade_file)
    data = bytearray(source.read_bytes())
    if len(data) < 8:
        raise ValueError("FIELD SHD 文件过短。")
    record_bytes = 4 * struct.unpack_from("<i", data, 0)[0]
    if record_bytes <= 0 or len(data) < 2 * record_bytes:
        raise ValueError("FIELD SHD 记录长度无效。")
    marker = bytes(data[record_bytes:record_bytes + 10]).strip(b"\x00 ")
    if marker:
        raise ValueError("SHD PlotType 不是旧版空标记，无法按兼容格式读取。")
    data[record_bytes:record_bytes + 10] = b"rectilin  "
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".shd", delete=False) as stream:
            stream.write(data)
            temporary_path = Path(stream.name)
        return read_shd_bin(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_field_result(shade_file: str | Path) -> Dict[str, Any]:
    """读取 FIELD 生成的 SHD 文件，并自动兼容新旧 PlotType 格式。"""
    try:
        result = read_shd_bin(shade_file)
    except ValueError as exc:
        if "PlotType" not in str(exc):
            raise
        result = _read_legacy_field_shd(shade_file)
    tl = np.asarray(result["tl"], dtype=np.float64)
    if tl.ndim != 4:
        raise ValueError(f"Kraken SHD 的 TL 维度应为 4，实际为 {tl.ndim}。")
    return result


@contextmanager
def _netcdf_writer(path: str | Path):
    """在 Windows 中文路径下通过 ASCII 临时文件安全写出 NetCDF。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    needs_staging = os.name == "nt" and not str(target).isascii()
    if not needs_staging:
        with netCDF4.Dataset(str(target), "w", format="NETCDF4") as dataset:
            yield dataset
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix="kraken_netcdf_", suffix=".nc")
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        with netCDF4.Dataset(str(temporary), "w", format="NETCDF4") as dataset:
            yield dataset
        shutil.copyfile(temporary, partial)
        os.replace(partial, target)
    finally:
        temporary.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def write_tl_netcdf(result: Mapping[str, Any], output_file: str | Path) -> Path:
    """把 FIELD 结果写成统一后端和前端可读取的 NetCDF 文件。"""
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    tl = np.asarray(result["tl"], dtype=np.float64)
    theta_count, source_count, depth_count, range_count = tl.shape
    with _netcdf_writer(target) as dataset:
        dataset.title = "Kraken normal-mode transmission loss"
        dataset.calculation_model = "Kraken + FIELD"
        dataset.source_file_format = "Acoustics Toolbox SHD"
        dataset.createDimension("theta", theta_count)
        dataset.createDimension("source_depth", source_count)
        dataset.createDimension("receiver_depth", depth_count)
        dataset.createDimension("range", range_count)
        theta = dataset.createVariable("theta_deg", "f4", ("theta",))
        source_depth = dataset.createVariable("source_depth_m", "f4", ("source_depth",))
        receiver_depth = dataset.createVariable("receiver_depth_m", "f4", ("receiver_depth",))
        ranges = dataset.createVariable("range_km", "f4", ("range",))
        transmission_loss = dataset.createVariable(
            "tl_db", "f4", ("theta", "source_depth", "receiver_depth", "range"),
            zlib=True, complevel=4, fill_value=np.float32(np.nan),
        )
        theta.units = "degree"
        source_depth.units = "m"
        receiver_depth.units = "m"
        ranges.units = "km"
        transmission_loss.units = "dB"
        theta[:] = np.asarray(result["theta_deg"], dtype=np.float32)
        source_depth[:] = np.asarray(result["source_depth_m"], dtype=np.float32)
        receiver_depth[:] = np.asarray(result["receiver_depth_m"], dtype=np.float32)
        ranges[:] = np.asarray(result["range_km"], dtype=np.float32)
        transmission_loss[:] = tl.astype(np.float32)
    return target


def field_statistics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """计算声场坐标范围和有限 TL 值的简要统计。"""
    tl = np.asarray(result["tl"], dtype=np.float64)
    finite = tl[np.isfinite(tl)]
    ranges = np.asarray(result["range_km"], dtype=np.float64)
    depths = np.asarray(result["receiver_depth_m"], dtype=np.float64)
    return {
        "frequency_hz": float(result["freq_hz"]),
        "source_depth_count": int(tl.shape[1]),
        "receiver_depth_count": int(tl.shape[2]),
        "receiver_range_count": int(tl.shape[3]),
        "range_min_km": float(np.nanmin(ranges)),
        "range_max_km": float(np.nanmax(ranges)),
        "depth_min_m": float(np.nanmin(depths)),
        "depth_max_m": float(np.nanmax(depths)),
        "tl_min_db": float(np.nanmin(finite)) if finite.size else math.nan,
        "tl_max_db": float(np.nanmax(finite)) if finite.size else math.nan,
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """以 UTF-8 编码写出带中文的 JSON 摘要文件。"""
    target = Path(path)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
