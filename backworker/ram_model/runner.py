# -*- coding: utf-8 -*-
"""RAM 输入准备、子进程执行、结果校验和导出。"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .config import load_config
from .environment import write_ram_input
from .results import read_tl_grid, read_tl_line, result_statistics, write_json, write_tl_netcdf


ProgressCallback = Callable[[str, str], None]


class RamCalculationError(RuntimeError):
    """表示 RAM 计算失败且错误可直接反馈给调用方。"""


def _silent_progress(stage: str, message: str) -> None:
    """未提供进度回调时使用的空函数。"""
    del stage, message


def _run_program(executable: Path, work_dir: Path, timeout_seconds: int) -> None:
    """在独立任务目录运行固定读取 `ram.in` 的 RAM 程序。"""
    command = [str(executable)]
    (work_dir / "ram.command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
    try:
        completed = subprocess.run(
            command,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RamCalculationError(f"RAM 运行超过 {timeout_seconds} 秒。") from exc
    (work_dir / "ram.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (work_dir / "ram.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    combined = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    upper = combined.upper()
    has_problem = any(token in upper for token in ("FORTRAN RUNTIME ERROR", "FATAL", "SEGMENTATION FAULT"))
    if completed.returncode != 0 or has_problem:
        detail = combined or "没有错误详情"
        raise RamCalculationError(f"RAM 计算失败（返回码 {completed.returncode}）：{detail[-2000:]}")


class RamRunner:
    """管理一次 RAM 计算的输入、执行和结果文件。"""

    def __init__(self, config: Mapping[str, Any]):
        """保存已经校验并解析路径的 RAM 配置。"""
        self.config = dict(config)

    def _make_work_dir(self, output_dir: Optional[str | Path]) -> Path:
        """创建不会与其他 RAM 任务冲突的独立目录。"""
        base = Path(output_dir).resolve() if output_dir else Path(self.config["OUTPUT"]["DIRECTORY"])
        base.mkdir(parents=True, exist_ok=True)
        work_dir = base / f"ram_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=False, exist_ok=False)
        return work_dir

    def generate_input(self, work_dir: str | Path) -> str:
        """只生成 RAM 固定名称输入文件，不启动求解器。"""
        directory = Path(work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return str(write_ram_input(directory / "ram.in", self.config))

    def run(
        self,
        output_dir: Optional[str | Path] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """执行 RAM、解析二维 TL 和接收深度曲线，并按配置导出 NetCDF。"""
        callback = progress or _silent_progress
        work_dir = self._make_work_dir(output_dir)
        input_file = self.generate_input(work_dir)
        executable = Path(self.config["EXECUTABLES"]["RAM"])
        if not executable.is_file():
            raise FileNotFoundError(f"RAM 可执行文件不存在：{executable}")
        callback("ram", "正在执行 RAM 抛物方程传播计算。")
        _run_program(executable, work_dir, int(self.config["RUNTIME"]["TIMEOUT_SECONDS"]))
        grid_file = work_dir / "tl.grid"
        line_file = work_dir / "tl.line"
        if not grid_file.is_file() or grid_file.stat().st_size == 0:
            raise RamCalculationError("RAM 未生成有效的 tl.grid。")
        if not line_file.is_file() or line_file.stat().st_size == 0:
            raise RamCalculationError("RAM 未生成有效的 tl.line。")
        grid = read_tl_grid(grid_file)
        line = read_tl_line(line_file)
        result: Dict[str, Any] = {
            "status": "field_ready",
            "work_dir": str(work_dir),
            "config_file": str(self.config.get("_CONFIG_FILE", "")),
            "input_file": input_file,
            "grid_file": str(grid_file),
            "line_file": str(line_file),
            "statistics": result_statistics(grid, line),
        }
        if bool(self.config["OUTPUT"].get("WRITE_NETCDF", True)):
            result["netcdf_file"] = str(write_tl_netcdf(grid, line, work_dir / "ram_tl.nc"))
        summary_file = work_dir / "result_summary.json"
        result["summary_file"] = str(summary_file)
        write_json(summary_file, result)
        callback("done", "RAM 计算完成。")
        return result


def run_ram(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """供统一任务分发层调用的一站式 RAM 配置加载和执行函数。"""
    return RamRunner(load_config(config_path, overrides)).run(output_dir=output_dir, progress=progress)
