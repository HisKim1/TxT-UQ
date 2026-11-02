from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_full_launcher_routes_all_stages_with_relocated_paths(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args.pop(0) == "run"
if args[:2] == ["python", "-c"]:
    os.execv(sys.executable, [sys.executable, *args[1:]])
if args[:2] == ["python", "scripts/run_gpu_jobs.py"]:
    for command in args[args.index("--") + 1:]:
        import shlex
        import subprocess
        command = command.replace("{gpu}", "0")
        subprocess.run(["bash", "-c", command], check=True)
    raise SystemExit(0)
with open(os.environ["CALL_LOG"], "a") as handle:
    handle.write(json.dumps(args) + "\n")
if args[0].endswith("build_reference_cache.py"):
    cache = Path(args[args.index("--cache-dir") + 1])
    cache.mkdir(parents=True)
    (cache / "manifest.json").write_text("{}")
if args[0].endswith("train_txt.py"):
    (Path(os.environ["TXT_RUN_ROOT"]) / (args[args.index("--var") + 1] + ".trained")).touch()
if args[0].endswith("validate_reference_checkpoint.py"):
    marker = Path(os.environ["TXT_RUN_ROOT"]) / (args[args.index("--var") + 1] + ".trained")
    raise SystemExit(0 if marker.exists() else 1)
assert args[0] == "pytest" or Path(args[0]).is_file(), args
"""
    )
    fake_uv.chmod(493)
    run = tmp_path / "saved run"
    calls = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        GPUS="0,1",
        TXT_DATA_ROOT=str(tmp_path / "input data"),
        TXT_RUN_ROOT=str(run),
        CALL_LOG=str(calls),
    )
    env.pop("TXT_CACHE_ROOT", None)
    subprocess.run(
        ["bash", str(ROOT / "scripts/run_txt_uq.sh"), "--execute"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    trained = [args for args in commands if args[0].endswith("train_txt.py")]
    assert len(trained) == 2
    for args in trained:
        assert args[args.index("--cache-dir") + 1].startswith(str(run / "cache"))
    sampled = [args for args in commands if args[0].endswith("sample_txt_uq.py")]
    assert sampled
    assert {args[args.index("--output-tag") + 1] for args in sampled} == {"txt_uq_k5"}
    for stage in ["train_txt_uq.py", "compute_residuals.py", "sample_txt_uq.py"]:
        assert any((args[0].endswith(stage) for args in commands)), stage
    for name in ["txt_uq.done"]:
        assert (run / "logs" / name).read_text().startswith("exit_code=0")
