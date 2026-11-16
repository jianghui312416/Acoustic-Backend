# -*- coding: utf-8 -*-
"""Bellhop 独立配置的读取、覆盖合并、路径解析与参数校验。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = MODULE_DIR.parent.parent / "config" / "bellhop_config.json"


def _deep_merge(target: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    """递归合并配置覆盖值，并避免修改调用方提供的对象。"""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _resolve_path(value: Any, config_file: Path) -> str:
    """把相对路径按配置文件所在目录转换为绝对路径。"""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return str(path.resolve())


def _finite(value: Any, name: str) -> float:
    """读取有限浮点数，并在错误信息中标出配置字段。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def _positive_integer(value: Any, name: str) -> int:
    """读取严格为正的整数配置。"""
    number = _finite(value, name)
    integer = int(round(number))
    if integer <= 0 or abs(number - integer) > 1.0e-9:
        raise ValueError(f"{name} 必须是正整数。")
    return integer


def validate_config(config: Mapping[str, Any]) -> None:
    """校验 Bellhop 求解器、网格、边界和海底参数配置。"""
    environment = config.get("BELLHOP_ENV")
    bottom_geo = config.get("BOTTOM_GEOACOUSTICS")
    if not isinstance(environment, Mapping) or not isinstance(bottom_geo, Mapping):
        raise ValueError("Bellhop 配置必须包含 BELLHOP_ENV 和 BOTTOM_GEOACOUSTICS 对象。")
    for section_name in ("SSP", "SURFACE", "BOTTOM"):
        if not isinstance(environment.get(section_name), Mapping):
            raise ValueError(f"BELLHOP_ENV.{section_name} 必须是对象。")
    if not str(config.get("BELLHOP_CXX") or "").strip():
        raise ValueError("BELLHOP_CXX 不能为空。")
    if not str(config.get("BELLHOP_CXX3d") or "").strip():
        raise ValueError("BELLHOP_CXX3d 不能为空。")
    for key in ("PADDING_KM", "BOX_GUARD_KM", "TRACE_EXTRA_KM", "STEP_M", "MAX_WAIT_TIME"):
        if _finite(config.get(key), key) <= 0.0:
            raise ValueError(f"{key} 必须为正数。")
    for key in ("N_ALPHA", "N_RANGE", "N_RD", "TL_N_RANGE", "TL_N_RD", "MAX_2D_WORKERS"):
        _positive_integer(config.get(key), key)
    if any(int(config[key]) < 2 for key in ("N_RANGE", "N_RD", "TL_N_RANGE", "TL_N_RD")):
        raise ValueError("N_RANGE、N_RD、TL_N_RANGE 和 TL_N_RD 至少为 2。")
    if _positive_integer(config.get("N_2D_PROFILES"), "N_2D_PROFILES") < 3:
        raise ValueError("N_2D_PROFILES 至少为 3。")
    ssp_mode = str(environment["SSP"].get("MODE", "range_dependent")).strip().lower()
    if ssp_mode not in {"fixed", "mean", "range_dependent"}:
        raise ValueError("BELLHOP_ENV.SSP.MODE 只能是 fixed、mean 或 range_dependent。")
    surface = environment["SURFACE"]
    if str(surface.get("BOUNDARY_CONDITION", "V")).upper() not in {"V", "R"}:
        raise ValueError("BELLHOP_ENV.SURFACE.BOUNDARY_CONDITION 只能是 V 或 R。")
    if str(surface.get("ATTENUATION_UNITS", "W")).upper() not in {"N", "F", "M", "W", "Q", "L"}:
        raise ValueError("BELLHOP_ENV.SURFACE.ATTENUATION_UNITS 配置无效。")
    bottom = environment["BOTTOM"]
    if str(bottom.get("BOUNDARY_CONDITION", "A")).upper() != "A":
        raise ValueError("本项目要求 BELLHOP_ENV.BOTTOM.BOUNDARY_CONDITION 为 A。")
    if _finite(bottom.get("COMPRESSIONAL_SPEED_MPS"), "海底纵波声速") <= 0.0:
        raise ValueError("海底纵波声速必须为正数。")
    if _finite(bottom.get("DENSITY_G_CM3"), "海底密度") <= 0.0:
        raise ValueError("海底密度必须为正数。")


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取 Bellhop 分类配置，应用覆盖值并解析求解器路径。"""
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_FILE
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Bellhop 配置文件根节点必须是对象。")
    if overrides:
        _deep_merge(config, overrides)
    config["BELLHOP_CXX"] = _resolve_path(config.get("BELLHOP_CXX", ""), path)
    config["BELLHOP_CXX3d"] = _resolve_path(config.get("BELLHOP_CXX3d", ""), path)
    config["_CONFIG_FILE"] = str(path)
    validate_config(config)
    return config
