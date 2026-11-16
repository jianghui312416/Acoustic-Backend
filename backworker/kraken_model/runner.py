# -*- coding: utf-8 -*-
"""Kraken 与 FIELD 子进程执行器。"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .config import load_config
from .environment import write_environment_file, write_field_file
from .results import field_statistics, load_field_result, parse_mode_summary, write_json, write_tl_netcdf


ProgressCallback = Callable[[str, str], None]


class KrakenCalculationError(RuntimeError):
    """表示 Kraken 或 FIELD 计算失败且错误可直接反馈给调用方。"""


def _silent_progress(stage: str, message: str) -> None:
    """未传进度回调时使用的空实现。"""
    del stage, message


def _print_problem(print_file: Path) -> str:
    """扫描 Acoustics Toolbox 打印文件中的致命错误信息。"""
    if not print_file.is_file():
        return ""
    text = print_file.read_text(encoding="utf-8", errors="replace")
    problem_lines = []
    for line in text.splitlines():
        upper = line.upper()
        if "FATAL" in upper or "ERROR" in upper or "FAILURE" in upper:
            problem_lines.append(line.strip())
    return "；".join(problem_lines[-8:])


def _program_print_file(work_dir: Path, root_name: str, label: str) -> Path:
    """返回不同 Acoustics Toolbox 程序实际使用的打印文件路径。"""
    if label.lower() == "field":
        return work_dir / "field.prt"
    return work_dir / f"{root_name}.prt"


def _run_program(executable: Path, root_name: str, work_dir: Path, timeout_seconds: int, label: str) -> None:
    """运行一个 Acoustics Toolbox 程序并保存标准输出、错误输出和命令记录。"""
    command = [str(executable), root_name]
    (work_dir / f"{root_name}.{label}.command.txt").write_text(
        subprocess.list2cmdline(command), encoding="utf-8"
    )
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
        raise KrakenCalculationError(f"{label} 运行超过 {timeout_seconds} 秒。") from exc
    (work_dir / f"{root_name}.{label}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (work_dir / f"{root_name}.{label}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    print_problem = _print_problem(_program_print_file(work_dir, root_name, label))
    if completed.returncode != 0 or print_problem:
        detail = print_problem or (completed.stderr or completed.stdout or "没有错误详情").strip()
        raise KrakenCalculationError(f"{label} 计算失败（返回码 {completed.returncode}）：{detail[-2000:]}")


class KrakenRunner:
    """负责准备文件、执行 Kraken/FIELD 并整理结果的独立运行器。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        profiles: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        """保存 Kraken 配置及可选的沿程环境剖面。

        每个剖面需要包含 ``range_km`` 和已校验的 ``config``。多剖面
        会依次写入同一个 Kraken 主环境文件，供求解器一次生成
        包含所有位置模态的 `.mod` 文件。
        """
        self.config = dict(config)
        self.profiles = list(profiles or [])

    def _make_work_dir(self, output_dir: Optional[str | Path]) -> Path:
        """建立仅包含 ASCII 算例文件名的独立任务目录。"""
        base = Path(output_dir).resolve() if output_dir else Path(self.config["OUTPUT"]["DIRECTORY"])
        base.mkdir(parents=True, exist_ok=True)
        task_dir = base / f"kraken_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        task_dir.mkdir(parents=False, exist_ok=False)
        return task_dir

    def _normalised_profiles(self) -> list[Dict[str, Any]]:
        """整理多剖面输入，并检查距离轴是否符合 FIELD 要求。"""
        if not self.profiles:
            return [{"range_km": 0.0, "config": self.config}]
        result = []
        for index, item in enumerate(self.profiles, 1):
            if "range_km" not in item or "config" not in item:
                raise ValueError(f"Kraken 第 {index} 个剖面缺少 range_km 或 config。")
            result.append({"range_km": float(item["range_km"]), "config": item["config"]})
        ranges = [item["range_km"] for item in result]
        if abs(ranges[0]) > 1.0e-9:
            raise ValueError("Kraken 第一个环境剖面必须从 0 km 开始。")
        if any(right <= left for left, right in zip(ranges, ranges[1:])):
            raise ValueError("Kraken 环境剖面距离必须严格递增。")
        return result

    def generate_inputs(self, work_dir: str | Path, root_name: str = "kraken_case") -> Dict[str, Any]:
        """生成 Kraken/FIELD 输入，保留每个分段 `.env` 供检查。"""
        directory = Path(work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        profiles = self._normalised_profiles()
        profile_files = []
        blocks = []
        for index, item in enumerate(profiles, 1):
            profile_file = write_environment_file(
                directory / f"{root_name}_p{index:03d}.env",
                item["config"],
            )
            profile_files.append(profile_file)
            blocks.append(profile_file.read_text(encoding="ascii"))

        # Kraken 的距离相关输入是把多个完整环境块顺序放入同一主文件。
        env_file = directory / f"{root_name}.env"
        env_file.write_text("".join(blocks), encoding="ascii")
        top_reflection_file: Path | None = None
        environment = self.config["ENVIRONMENT"]
        if str(environment.get("TOP_BOUNDARY", "V")).upper() == "F":
            reflection_source = Path(str(environment["TOP_REFLECTION_FILE"])).resolve()
            if not reflection_source.is_file():
                raise FileNotFoundError(f"Kraken 顶边界反射系数文件不存在：{reflection_source}")
            top_reflection_file = env_file.with_suffix(".trc")
            if reflection_source != top_reflection_file.resolve():
                shutil.copy2(reflection_source, top_reflection_file)
        profile_ranges = [float(item["range_km"]) for item in profiles]
        flp_file = write_field_file(
            directory / f"{root_name}.flp",
            self.config,
            profile_ranges_km=profile_ranges,
        )
        return {
            "environment_file": str(env_file),
            "environment_files": [str(path) for path in profile_files],
            "field_file": str(flp_file),
            "top_reflection_file": (
                str(top_reflection_file) if top_reflection_file is not None else None
            ),
            "profile_ranges_km": profile_ranges,
            "profile_count": len(profiles),
        }

    def run(
        self,
        output_dir: Optional[str | Path] = None,
        run_field: Optional[bool] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """执行完整的 Kraken 模态计算，并按配置选择是否继续计算 TL 声场。"""
        callback = progress or _silent_progress
        work_dir = self._make_work_dir(output_dir)
        root_name = "kraken_case"
        input_files = self.generate_inputs(work_dir, root_name)
        timeout = int(self.config["RUNTIME"]["TIMEOUT_SECONDS"])
        kraken_executable = Path(self.config["EXECUTABLES"]["KRAKEN"])
        if not kraken_executable.is_file():
            raise FileNotFoundError(f"Kraken 可执行文件不存在：{kraken_executable}")

        callback("kraken", "正在计算 Kraken 简正模态。")
        _run_program(kraken_executable, root_name, work_dir, timeout, "kraken")
        mode_file = work_dir / f"{root_name}.mod"
        print_file = work_dir / f"{root_name}.prt"
        if not mode_file.is_file() or mode_file.stat().st_size == 0:
            raise KrakenCalculationError("Kraken 未生成有效的 .mod 模态文件。")
        kraken_print = work_dir / f"{root_name}.kraken.prt"
        shutil.copy2(print_file, kraken_print)
        mode_summary = parse_mode_summary(kraken_print)

        should_run_field = bool(self.config["RUNTIME"].get("RUN_FIELD", True)) if run_field is None else bool(run_field)
        result: Dict[str, Any] = {
            "status": "modes_ready",
            "work_dir": str(work_dir),
            "config_file": str(self.config.get("_CONFIG_FILE", "")),
            "environment_file": input_files["environment_file"],
            "environment_files": input_files["environment_files"],
            "field_file": input_files["field_file"],
            "top_reflection_file": input_files["top_reflection_file"],
            "profile_ranges_km": input_files["profile_ranges_km"],
            "profile_count": input_files["profile_count"],
            "mode_file": str(mode_file),
            "kraken_print_file": str(kraken_print),
            "mode_summary": mode_summary,
        }
        if should_run_field:
            field_executable = Path(self.config["EXECUTABLES"]["FIELD"])
            if not field_executable.is_file():
                raise FileNotFoundError(f"FIELD 可执行文件不存在：{field_executable}")
            callback("field", "正在使用 FIELD 叠加模态并生成传播损失。")
            _run_program(field_executable, root_name, work_dir, timeout, "field")
            shade_file = work_dir / f"{root_name}.shd"
            field_print = work_dir / f"{root_name}.field.prt"
            native_field_print = _program_print_file(work_dir, root_name, "field")
            if native_field_print.is_file():
                shutil.copy2(native_field_print, field_print)
            if not shade_file.is_file() or shade_file.stat().st_size == 0:
                raise KrakenCalculationError("FIELD 未生成有效的 .shd 声场文件。")
            field_result = load_field_result(shade_file)
            result.update({
                "status": "field_ready",
                "shade_file": str(shade_file),
                "field_print_file": str(field_print),
                "field_statistics": field_statistics(field_result),
            })
            if bool(self.config["OUTPUT"].get("WRITE_NETCDF", True)):
                netcdf_file = write_tl_netcdf(field_result, work_dir / "kraken_tl.nc")
                result["netcdf_file"] = str(netcdf_file)
        callback("done", "Kraken 计算完成。")
        summary_file = work_dir / "result_summary.json"
        result["summary_file"] = str(summary_file)
        write_json(summary_file, result)
        return result


def run_kraken(
    overrides: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    run_field: Optional[bool] = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """供统一任务分发层调用：加载配置、应用覆盖参数并执行 Kraken。"""
    config = load_config(config_path, overrides)
    return KrakenRunner(config).run(output_dir=output_dir, run_field=run_field, progress=progress)
