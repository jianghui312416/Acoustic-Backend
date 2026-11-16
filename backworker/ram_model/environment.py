# -*- coding: utf-8 -*-
"""生成 RAM 1.5 使用的固定名称输入文件 `ram.in`。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def _safe_title(value: Any) -> str:
    """将标题转换成旧版 Fortran 稳定支持的 ASCII 字符串。"""
    title = str(value or "RAM calculation").encode("ascii", errors="replace").decode("ascii")
    return " ".join(title.split()) or "RAM calculation"


def _pair_lines(values: Iterable[Iterable[Any]]) -> list[str]:
    """把二元数值数组转换为 RAM 的逐行文本。"""
    return [f"{float(row[0]):.10g} {float(row[1]):.10g}" for row in values]


def write_ram_input(path: str | Path, config: Mapping[str, Any]) -> Path:
    """写出包含网格、海底地形和一个或多个距离剖面的 `ram.in`。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = config["MODEL"]
    environment = config["ENVIRONMENT"]
    lines = [
        _safe_title(model.get("TITLE")),
        (
            f"{float(model['FREQUENCY_HZ']):.10g} "
            f"{float(model['SOURCE_DEPTH_M']):.10g} "
            f"{float(model['RECEIVER_DEPTH_M']):.10g}"
        ),
        (
            f"{float(model['MAX_RANGE_M']):.10g} "
            f"{float(model['RANGE_STEP_M']):.10g} "
            f"{int(model['RANGE_OUTPUT_DECIMATION'])}"
        ),
        (
            f"{float(model['MAX_DEPTH_M']):.10g} "
            f"{float(model['DEPTH_STEP_M']):.10g} "
            f"{int(model['DEPTH_OUTPUT_DECIMATION'])} "
            f"{float(model['OUTPUT_DEPTH_MAX_M']):.10g}"
        ),
        (
            f"{float(model['REFERENCE_SOUND_SPEED_MPS']):.10g} "
            f"{int(model['PADE_TERMS'])} "
            f"{int(model['STABILITY_CONSTRAINTS'])} "
            f"{float(model.get('STABILITY_DISABLE_RANGE_M', 0.0)):.10g}"
        ),
    ]
    lines.extend(_pair_lines(environment["BATHYMETRY"]))
    lines.append("-1 -1")
    profile_keys = (
        "WATER_SOUND_SPEED_MPS",
        "BOTTOM_SOUND_SPEED_MPS",
        "BOTTOM_DENSITY_G_CM3",
        "BOTTOM_ATTENUATION_DB_PER_WAVELENGTH",
    )
    for index, profile in enumerate(environment["PROFILES"]):
        if index > 0:
            lines.append(f"{float(profile['START_RANGE_M']):.10g}")
        for key in profile_keys:
            lines.extend(_pair_lines(profile[key]))
            lines.append("-1 -1")
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target
