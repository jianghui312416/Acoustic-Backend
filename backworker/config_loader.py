# -*- coding: utf-8 -*-
"""读取并合并后端的数据、Bellhop、Kraken 和 RAM 四类配置。"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from backworker.bellhop_model.config import load_config as load_bellhop_config


PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
CONFIG_FILES = {
    "data": CONFIG_DIR / "data_config.json",
    "bellhop": CONFIG_DIR / "bellhop_config.json",
    "kraken": CONFIG_DIR / "kraken_config.json",
    "ram": CONFIG_DIR / "ram_config.json",
}


def _read_json(path: Path) -> Dict[str, Any]:
    """读取一份 UTF-8 JSON 配置并校验根节点类型。"""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件根节点必须是对象：{path}")
    return value


def _merge(target: Dict[str, Any], source: Mapping[str, Any], source_name: str) -> None:
    """合并顶层配置，并拒绝不同文件意外定义同名业务字段。"""
    for key, value in source.items():
        if key.startswith("_"):
            continue
        if key in target:
            raise ValueError(f"配置字段 {key} 在多个文件中重复定义，冲突来源：{source_name}")
        target[key] = copy.deepcopy(value)


def load_backend_config() -> Dict[str, Any]:
    """合并公共数据配置和 Bellhop 配置，并挂载 Kraken/RAM 独立配置。"""
    data = _read_json(CONFIG_FILES["data"])
    # Bellhop 与 Kraken、RAM 一样，通过模型子包完成独立校验和路径解析。
    bellhop = load_bellhop_config(CONFIG_FILES["bellhop"])
    kraken = _read_json(CONFIG_FILES["kraken"])
    ram = _read_json(CONFIG_FILES["ram"])
    merged: Dict[str, Any] = {}
    _merge(merged, data, "data_config.json")
    _merge(merged, bellhop, "bellhop_config.json")
    merged["KRAKEN_CONFIG"] = kraken
    merged["RAM_CONFIG"] = ram
    merged["CONFIG_FILES"] = {name: str(path) for name, path in CONFIG_FILES.items()}
    if not isinstance(merged.get("DATA_SOURCES"), Mapping):
        raise ValueError("data_config.json 缺少 DATA_SOURCES。")
    if not isinstance(merged.get("DATA_SERVICES"), Mapping):
        raise ValueError("data_config.json 缺少 DATA_SERVICES。")
    api0 = merged["DATA_SERVICES"].get("api0")
    if isinstance(api0, dict) and not str(api0.get("token") or "").strip():
        token_env = str(api0.get("token_env") or "OCEAN_API0_TOKEN").strip()
        api0["token"] = os.environ.get(token_env, "")
    return merged


def config_mtime_snapshot() -> tuple[int, ...]:
    """返回四份配置文件的修改时间快照，用于请求间热加载。"""
    return tuple(path.stat().st_mtime_ns for path in CONFIG_FILES.values())
