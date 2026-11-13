from types import SimpleNamespace


DATASETS = {
    "mnist": {
        "seed": 88,
        "num_tasks": 5,
        "num_classes": 10,
        "num_classes_per_task": 2,
        "n_steps": 5,
        "task_epochs": (8, 8, 8, 8, 8),
        "learning_rate": 0.005,
    },
    "fashionmnist": {
        "seed": 201,
        "num_tasks": 5,
        "num_classes": 10,
        "num_classes_per_task": 2,
        "n_steps": 5,
        "task_epochs": (8, 8, 8, 8, 8),
        "learning_rate": 0.005,
    },
    "cifar10": {
        "seed": 88,
        "num_tasks": 5,
        "num_classes": 10,
        "num_classes_per_task": 2,
        "n_steps": 5,
        "task_epochs": (8, 8, 8, 8, 8),
        "learning_rate": 0.005,
    },
    "cifar100": {
        "seed": 123,
        "num_tasks": 10,
        "num_classes": 100,
        "num_classes_per_task": 10,
        "n_steps": 9,
        "task_epochs": (100,) * 10,
        "learning_rate": 0.001,
    },
}

METHODS = {
    "encl": {
        "learner": "encl",
        "arch": "vgg9_tess",
        "training_mode": "tess",
    },
    "bptt_noncl": {
        "learner": "bptt_noncl",
        "arch": "vgg9_bptt",
        "training_mode": "bptt",
    },
    "tess_noncl": {
        "learner": "tess_noncl",
        "arch": "vgg9_tess",
        "training_mode": "tess",
    },
    "mr": {"learner": "mr"},
}

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
IDLE_BASELINE_SECONDS = 20.0


def build_config(
    method,
    dataset,
    data_path,
    output_dir,
    gpu_index,
    measure_gpu_index,
):
    data = DATASETS[dataset]
    values = {
        "method": method,
        "dataset": dataset,
        "data_path": data_path,
        "output_dir": output_dir,
        "gpu_index": gpu_index,
        "measure_gpu_index": measure_gpu_index,
        "batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": TEST_BATCH_SIZE,
        "seed": data["seed"],
        "num_tasks": data["num_tasks"],
        "num_classes": data["num_classes"],
        "num_classes_per_task": data["num_classes_per_task"],
        "n_steps": data["n_steps"],
        "task_epochs": data["task_epochs"],
        "epochs": data["task_epochs"][0],
    }
    if method == "mr":
        values.update(
            lr=data["learning_rate"],
            idle_seconds=IDLE_BASELINE_SECONDS,
            replay_size=2000,
            replay_batch=48,
        )
    else:
        method_config = METHODS[method]
        values.update(
            arch=method_config["arch"],
            training_mode=method_config["training_mode"],
            lr_conv=data["learning_rate"],
            feature_head_lr=data["learning_rate"],
        )
        if method == "encl":
            values.update(
                sparse_init="uniform",
                mask_density=0.1 if dataset == "cifar100" else 0.2,
                max_overlap_fraction=0.01 if dataset == "cifar100" else 0.05,
                isolate=True,
                refill_overlap=True,
            )
    return SimpleNamespace(**values)
