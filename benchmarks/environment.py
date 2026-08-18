"""Environment preflight and metadata collection for Phase 3 benchmarks."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Iterable
from pathlib import Path

from benchmarks.results import EnvironmentMetadata

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _read_text(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8").strip()


def _read_first_match(path: str, pattern: str) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_cpu_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(int(cpu) for cpu in os.sched_getaffinity(0))


def _count_physical_cpus() -> int | None:
    cpuinfo = _read_text("/proc/cpuinfo")
    if cpuinfo is None:
        return None
    physical: set[tuple[str, str]] = set()
    current_physical = None
    current_core = None
    for line in cpuinfo.splitlines():
        if line.startswith("physical id"):
            current_physical = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            current_core = line.split(":", 1)[1].strip()
        elif line == "" and current_physical is not None and current_core is not None:
            physical.add((current_physical, current_core))
            current_physical = None
            current_core = None
    if current_physical is not None and current_core is not None:
        physical.add((current_physical, current_core))
    return len(physical) if physical else None


def _total_ram_bytes() -> int | None:
    mem_total = _read_first_match("/proc/meminfo", r"^MemTotal:\s+(\d+)\s+kB$")
    if mem_total is None:
        return None
    return int(mem_total) * 1024


def _is_wsl() -> bool:
    release = platform.release().lower()
    version = platform.version().lower()
    return (
        "microsoft" in release
        or "microsoft" in version
        or Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
    )


def _thread_env_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}


def apply_single_thread_env() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def single_thread_env() -> dict[str, str]:
    return {name: "1" for name in THREAD_ENV_VARS}


def worker_thread_env_snapshot() -> dict[str, str | None]:
    return _thread_env_snapshot()


def collect_environment() -> EnvironmentMetadata:
    warnings: list[str] = []
    is_wsl = _is_wsl()
    if is_wsl:
        warnings.append("WSL detected; official benchmark conditions are not fully controlled.")

    smt_status = _read_text("/sys/devices/system/cpu/smt/control")
    if smt_status is None:
        warnings.append("SMT status unavailable.")
    elif "off" not in smt_status.lower():
        warnings.append("SMT/hyperthreading does not appear to be disabled.")

    governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor is None:
        warnings.append("CPU governor unavailable.")
    elif governor != "performance":
        warnings.append(f"CPU governor is '{governor}', not 'performance'.")

    thp = _read_text("/sys/kernel/mm/transparent_hugepage/enabled")
    if thp is None:
        warnings.append("Transparent huge page state unavailable.")

    ready = not is_wsl and (
        smt_status is not None and "off" in smt_status.lower()
    ) and governor == "performance"

    return EnvironmentMetadata(
        cpu_model=(
            _read_first_match("/proc/cpuinfo", r"^model name\s+:\s+(.+)$")
            or platform.processor()
            or "unknown"
        ),
        logical_cpus=os.cpu_count(),
        physical_cpus=_count_physical_cpus(),
        cpu_affinity=_parse_cpu_affinity(),
        os=platform.system(),
        kernel=platform.release(),
        python_version=platform.python_version(),
        total_ram_bytes=_total_ram_bytes(),
        is_wsl=is_wsl,
        smt_status=smt_status,
        cpu_governor=governor,
        thp_state=thp,
        official_environment_ready=ready,
        warnings=warnings,
        env_threads=_thread_env_snapshot(),
    )


def ensure_official_mode_allowed(
    environment: EnvironmentMetadata, *, official: bool, allow_uncontrolled_environment: bool
) -> None:
    if not official:
        return
    if environment.official_environment_ready:
        return
    raise ValueError(
        "official mode requires OFFICIAL_ENVIRONMENT_READY=true; uncontrolled environments "
        "may only produce NON-OFFICIAL results"
    )


def environment_warnings_lines(environment: EnvironmentMetadata) -> Iterable[str]:
    for warning in environment.warnings:
        yield f"WARNING: {warning}"
