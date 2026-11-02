from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Mapping, Sequence

GPU_TOKEN = "{gpu}"


@dataclass
class RunningJob:
    process: subprocess.Popen[bytes]
    gpu: str
    index: int
    log_handle: object
    log_path: Path


def parse_gpus(value: str) -> list[str]:
    gpus = value.split(",")
    if not gpus or any((not re.fullmatch("\\d+", gpu) for gpu in gpus)):
        raise ValueError(f"invalid GPU list: {value!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU IDs must be unique: {value!r}")
    return gpus


def _signal_group(job: RunningJob, signum: signal.Signals) -> None:
    try:
        os.killpg(job.process.pid, signum)
    except ProcessLookupError:
        pass


def _stop_jobs(running: dict[int, RunningJob]) -> None:
    for job in running.values():
        _signal_group(job, signal.SIGTERM)
    for job in running.values():
        try:
            job.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _signal_group(job, signal.SIGKILL)
            job.process.wait()
        _signal_group(job, signal.SIGKILL)
        job.log_handle.close()


def run_fifo(
    commands: Sequence[str],
    gpus: Sequence[str],
    group: str,
    log_dir: Path,
    *,
    log_paths: Sequence[Path] | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    if not re.fullmatch("[A-Za-z0-9_.-]+", group):
        raise ValueError(f"unsafe group name: {group!r}")
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be nonempty and unique")
    if any((command.count(GPU_TOKEN) != 1 for command in commands)):
        raise ValueError(f"each job must contain exactly one {GPU_TOKEN!r} token")
    if log_paths is not None and len(log_paths) != len(commands):
        raise ValueError("log_paths must match the command count")
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = deque(enumerate(commands))
    available = deque(gpus)
    running: dict[int, RunningJob] = {}
    total = len(commands)
    completed = 0
    spawning = False
    cleaning = False
    pending_signal: int | None = None

    def stop_on_signal(signum: int, _frame: object) -> None:
        nonlocal pending_signal
        if cleaning:
            return
        if spawning:
            pending_signal = signum
            return
        raise SystemExit(128 + signum)

    managed_signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.signal(signum, stop_on_signal) for signum in managed_signals
    }
    print(f"Queued {total} FIFO jobs on GPUs {list(gpus)}", flush=True)
    try:
        while pending or running:
            while pending and available:
                index, template = pending.popleft()
                gpu = available.popleft()
                command = template.replace(GPU_TOKEN, gpu)
                log_path = (
                    Path(log_paths[index])
                    if log_paths is not None
                    else log_dir / f"{group}_job{index:03d}_g{gpu}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = None
                spawning = True
                try:
                    log_handle = log_path.open("wb")
                    process = subprocess.Popen(
                        ["bash", "-lc", f"exec {command}"],
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        cwd=cwd,
                        env=env,
                    )
                    running[process.pid] = RunningJob(
                        process, gpu, index, log_handle, log_path
                    )
                except BaseException:
                    if log_handle is not None:
                        log_handle.close()
                    raise
                finally:
                    spawning = False
                if pending_signal is not None:
                    signum = pending_signal
                    pending_signal = None
                    raise SystemExit(128 + signum)
                print(
                    f"START {group} job {index}/{total - 1} GPU {gpu}: {command}",
                    flush=True,
                )
            pid, status = os.waitpid(-1, 0)
            job = running.pop(pid)
            return_code = os.waitstatus_to_exitcode(status)
            job.process.returncode = return_code
            job.log_handle.close()
            if return_code != 0:
                print(
                    f"ERROR {group} job {job.index} GPU {job.gpu}; see {job.log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                running[pid] = job
                cleaning = True
                _stop_jobs(running)
                running.clear()
                return return_code if return_code > 0 else 128 - return_code
            _signal_group(job, signal.SIGKILL)
            completed += 1
            available.append(job.gpu)
            print(
                f"DONE {group} job {job.index} GPU {job.gpu} ({completed}/{total})",
                flush=True,
            )
    except BaseException:
        cleaning = True
        _stop_jobs(running)
        running.clear()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0
