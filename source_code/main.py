import argparse
import json
import os

from .common.experiment import print_and_log, setup
from .common.resource_meter import ResourceMeter
from .config import DATASETS, METHODS, build_config
from .data.snn_datamanager import SNNDataManager
from .methods.bptt_noncl import BPTTNoNCLLearner
from .methods.encl import ENCLLearner
from .methods.mr import run_experiment as run_mr
from .methods.tess_noncl import TESSNoNCLLearner


LEARNERS = {
    "encl": ENCLLearner,
    "bptt_noncl": BPTTNoNCLLearner,
    "tess_noncl": TESSNoNCLLearner,
}


def run_non_replay(args):
    setup(args)
    data_manager = SNNDataManager(args)
    meter = ResourceMeter(
        args.measure_gpu_index,
        os.path.join(args.output_dir, "resources.json"),
        detailed_sharing=args.dataset == "cifar100",
        emit_clean_fields=args.dataset == "cifar100",
        energy_exclusivity_required=(
            False
            if args.method == "encl"
            else args.dataset != "cifar100"
        ),
    )
    learner = LEARNERS[args.method](args, data_manager)
    learner._meter = meter
    for task, epochs in enumerate(args.task_epochs):
        print_and_log(f"Starting Task {task + 1}/{args.num_tasks}")
        learner.epochs = epochs
        learner.args.epochs = epochs
        learner._log_memory_usage(f"Task {task} - Start")
        learner.train()
        learner.after_task()
        learner._log_memory_usage(f"Task {task} - Complete")
    total_params = 0
    if learner.NET is not None:
        total_params += sum(
            parameter.numel() for parameter in learner.NET.parameters()
        )
    for stored in learner.task_heads.values():
        if isinstance(stored, dict):
            for value in stored.values():
                if hasattr(value, "numel"):
                    total_params += int(value.numel())
    resources = meter.finalize(
        model_params=total_params or None,
        model_mb=(
            total_params * 4.0 / 1_000_000.0
            if total_params
            else None
        ),
    )
    print_and_log(
        "[resource-meter] TRAIN-only resources: "
        + json.dumps(resources, indent=2)
    )
    learner.evaluate()
    learner.report_scores()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--measure-gpu-index", type=int, default=0)
    runtime = parser.parse_args()
    args = build_config(
        runtime.method,
        runtime.dataset,
        runtime.data_path,
        runtime.output_dir,
        runtime.gpu_index,
        runtime.measure_gpu_index,
    )
    if args.method == "mr":
        args.out = os.path.join(args.output_dir, "result.json")
        run_mr(args)
    else:
        args.save_path = os.path.join(args.output_dir, "exp")
        run_non_replay(args)


if __name__ == "__main__":
    main()
