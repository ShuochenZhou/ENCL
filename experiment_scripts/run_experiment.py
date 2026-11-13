#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = tuple(
    f"{dataset}/{method}"
    for dataset in ("split_mnist", "split_fmnist", "split_cifar10", "split_cifar100")
    for method in ("encl", "bptt_noncl", "tess_noncl", "mr")
)


def dataset_path(experiment: str, data_root: Path) -> Path:
    dataset = experiment.split("/", 1)[0]
    return {
        "split_mnist": data_root / "mnist",
        "split_fmnist": data_root / "fashionmnist",
        "split_cifar10": data_root / "cifar10",
        "split_cifar100": data_root / "cifar100",
    }[dataset]


def dataset_name(experiment: str) -> str:
    dataset = experiment.split("/", 1)[0]
    return {
        "split_mnist": "mnist",
        "split_fmnist": "fashionmnist",
        "split_cifar10": "cifar10",
        "split_cifar100": "cifar100",
    }[dataset]


def build_command(experiment: str, args, output: Path) -> list[str]:
    method = experiment.split("/", 1)[1]
    data = dataset_path(experiment, args.data_root)
    dataset = dataset_name(experiment)
    return [
        args.python,
        "-u",
        "-m",
        "source_code.main",
        "--method",
        method,
        "--dataset",
        dataset,
        "--data-path",
        str(data),
        "--output-dir",
        str(output),
        "--gpu-index",
        "0",
        "--measure-gpu-index",
        str(args.gpu),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "outputs" / f"{args.experiment.replace('/', '_')}_{timestamp}"
    command = build_command(args.experiment, args, output)
    print(shlex.join(command))
    if args.print_only:
        return 0

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT) if not python_path else os.pathsep.join((str(ROOT), python_path))
    )
    with (output / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
