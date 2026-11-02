from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import LOGS_DIR
from gpu_scheduler import parse_gpus, run_fifo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path(LOGS_DIR))
    parser.add_argument("commands", nargs="+")
    args = parser.parse_args()
    try:
        return run_fifo(args.commands, parse_gpus(args.gpus), args.group, args.log_dir)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
