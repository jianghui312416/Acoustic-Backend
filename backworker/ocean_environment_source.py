# -*- coding: utf-8 -*-
"""配置驱动的公共海洋环境数据源实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from backworker.acoustic_environment import AcousticEnvironmentSource, get_bottom_geo_config
from backworker.bellhop_model import runner as legacy_environment


class ConfiguredOceanEnvironmentSource(AcousticEnvironmentSource):
    """从统一数据路由读取三个声学模型共用的海洋环境。"""

    def resolve_topography_file(self, params: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
        """解析本次任务使用的地形数据文件。"""
        value = (
            legacy_environment._resolve_project_path(params["topo_file"])
            if params.get("topo_file")
            else legacy_environment.resolve_configured_data_file(config, params, "topography")
        )
        return Path(str(value))

    def open_topography(self, path: Path):
        """打开公共地形数据读取器。"""
        return legacy_environment.TopographyReader(path)

    def build_source_grids(self, points, params, settings):
        """为全部声源建立公共环境采样网格。"""
        return legacy_environment._build_source_grids(points, params, settings)

    def open_data_access(self, params, config):
        """创建温盐、海况和海底数据访问对象。"""
        return legacy_environment._DataAccess(
            params,
            config,
            bottom_geo_config=get_bottom_geo_config(config),
        )

    def prepare_shared_environment(self, data_access, topography, source_grids, settings):
        """一次性读取并整理全部声源共享的环境数据。"""
        return legacy_environment._prepare_shared_source_environment(
            data_access,
            topography,
            source_grids,
            settings["depth_grid_m"],
            settings,
        )


DEFAULT_SOURCE = ConfiguredOceanEnvironmentSource()
