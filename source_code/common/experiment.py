import copy
import datetime
import json
import logging
import os
import random
import warnings
import numpy as np
import torch


def clear_gradients(parameters):
    for parameter in parameters:
        parameter.grad = None


@torch.no_grad()
def plain_sgd_update(parameters, learning_rate):
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not parameters:
        return
    torch._foreach_add_(
        parameters,
        [parameter.grad for parameter in parameters],
        alpha=-float(learning_rate),
    )

logger = None

def get_experiment_base_dir(args):
    return str(args.save_path).rstrip("/")

def print_and_log(message):
    print(message)
    if logger is not None:
        logger.info(message)

def setup(args):
    global logger
    args.experiment_folder = "{}_tsk{}_cls{}_{}_{}_{}".format(
        args.dataset,
        args.num_tasks,
        args.num_classes_per_task,
        args.method.replace("_", "-"),
        args.training_mode,
        args.arch.replace("_", "-"),
    )
    base = get_experiment_base_dir(args)
    root = os.path.join(base, args.experiment_folder)
    for name in ("models", "logs", "masks"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./masks", exist_ok=True)
    with open(os.path.join(root, "parameters.json"), "w") as handle:
        payload = copy.deepcopy(vars(args))
        payload["timestamp"] = datetime.datetime.now().isoformat()
        json.dump(payload, handle, indent=2, sort_keys=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    logger = logging.getLogger(args.experiment_folder)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(root, "logs", "run.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    print_and_log(args)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_labels(labels, task, classes_per_task):
    return labels - classes_per_task * task


def count_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return 0
