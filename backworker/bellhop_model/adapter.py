# -*- coding: utf-8 -*-
"""把后端公共任务字段整理为 Bellhop runner 使用的参数。"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional


FIELD_ALIASES = {
    "frequency_hz": "freq_hz",
    "source_depth": "source_depth_m",
    "max_range_km": "radius_km",
    "receiver_depth_m": "horizontal_tl_depth_m",
}


def build_overrides_from_backend(
    params: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """合并公共默认值、字段别名和请求参数，生成 Bellhop 任务参数。"""
    if not isinstance(params, Mapping):
        raise TypeError("Bellhop 任务参数必须是映射对象。")
    result: Dict[str, Any] = copy.deepcopy(dict(defaults or {}))
    for source_name, target_name in FIELD_ALIASES.items():
        if target_name not in params and source_name in params:
            result[target_name] = copy.deepcopy(params[source_name])
    result.update(copy.deepcopy(dict(params)))
    result["acoustic_model"] = "bellhop"
    if not result.get("selection_points") and result.get("center_lat") is not None and result.get("center_lon") is not None:
        result["selection_points"] = [{
            "lat": result["center_lat"],
            "lon": result["center_lon"],
        }]
    return result


def build_params_from_backend(
    params: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """提供语义更直观的别名，兼容后续调用方逐步迁移。"""
    return build_overrides_from_backend(params, defaults)
