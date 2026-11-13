import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


CIFAR100_MEAN = tuple(value / 255 for value in (125.3, 123.0, 113.9))
CIFAR100_STD = tuple(value / 255 for value in (63.0, 62.1, 66.7))


class SplitTorchvisionDataset:
    def __init__(self, dataset_class, root, train, transform, tasks):
        self.data = dataset_class(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )
        self.task_id = 0
        self.split_datasets = [[] for _ in tasks]
        for index, (_, label) in enumerate(self.data):
            for task_id, task_labels in enumerate(tasks):
                if label in task_labels:
                    self.split_datasets[task_id].append(index)

    def set_task(self, task_id):
        self.task_id = int(task_id)

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, index):
        return self.data[self.split_datasets[self.task_id][index]]


def build_torchvision_dataset(dataset, root, train, transform, tasks):
    if dataset == "mnist":
        return SplitTorchvisionDataset(
            datasets.MNIST, root, train, transform, tasks
        )
    if dataset == "fashionmnist":
        return SplitTorchvisionDataset(
            datasets.FashionMNIST, root, train, transform, tasks
        )
    if dataset == "cifar10":
        return SplitTorchvisionDataset(
            datasets.CIFAR10, root, train, transform, tasks
        )
    raise ValueError(f"unsupported torchvision dataset: {dataset}")


class BinarySplitCIFAR100(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = bool(train)
        self.task_id = 0
        self._x = None
        self._y = None
        self.set_task(0)

    def set_task(self, task_id):
        task_id = int(task_id)
        if self._x is not None and task_id == self.task_id:
            return
        split = "train" if self.train else "test"
        self.task_id = task_id
        self._x = torch.load(
            os.path.join(self.root, f"data{task_id}{split}x.bin"),
            map_location="cpu",
        ).float()
        local_y = torch.load(
            os.path.join(self.root, f"data{task_id}{split}y.bin"),
            map_location="cpu",
        ).long()
        self._y = local_y + task_id * 10

    def __len__(self):
        return int(self._y.numel())

    def __getitem__(self, index):
        return self._x[index], self._y[index]


def normalize_cifar100_uint8(images):
    contiguous = np.ascontiguousarray(images)
    tensor = torch.from_numpy(contiguous).permute(0, 3, 1, 2)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(CIFAR100_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std).contiguous()


def write_cifar100_split(images, targets, split, output):
    labels = np.asarray(targets, dtype=np.int64)
    for task in range(10):
        indices = np.flatnonzero(labels // 10 == task)
        task_images = normalize_cifar100_uint8(images[indices])
        task_labels = torch.from_numpy((labels[indices] % 10).copy()).long()
        expected_count = 5000 if split == "train" else 1000
        if len(indices) != expected_count:
            raise RuntimeError(
                f"unexpected {split} count for task {task}: "
                f"expected {expected_count}, observed {len(indices)}"
            )
        torch.save(task_images, output / f"data{task}{split}x.bin")
        torch.save(task_labels, output / f"data{task}{split}y.bin")


def build_cifar100_cache(
    data_root,
    output,
    download=False,
):
    data_root = Path(data_root)
    output = Path(output)
    if output.exists():
        raise RuntimeError(
            f"refusing to overwrite existing output: {output}; "
            "verify it or choose a different --output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        train = datasets.CIFAR100(
            root=str(data_root), train=True, download=download
        )
        test = datasets.CIFAR100(
            root=str(data_root), train=False, download=download
        )
        write_cifar100_split(train.data, train.targets, "train", staging)
        write_cifar100_split(test.data, test.targets, "test", staging)
        staging.rename(output)
        print(f"CIFAR100_CACHE_BUILD_PASS output={output}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ensure_cifar100_cache(data_root):
    data_root = Path(data_root)
    output = data_root / "binary_cifar100"
    expected = {
        output / f"data{task}{split}{kind}.bin"
        for task in range(10)
        for split in ("train", "test")
        for kind in ("x", "y")
    }
    existing = {path for path in expected if path.is_file()}
    if existing == expected:
        return output
    if existing or output.exists():
        missing = sorted(path.name for path in expected - existing)
        raise RuntimeError(
            f"incomplete CIFAR-100 cache at {output}; missing: {missing}"
        )
    try:
        build_cifar100_cache(data_root, output, download=True)
    except OSError:
        if not all(path.is_file() for path in expected):
            raise
    return output


def cifar100_data_cli():
    parser = argparse.ArgumentParser(
        description="Build the normalized Split CIFAR-100 cache"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path)
    build.add_argument("--download", action="store_true")

    args = parser.parse_args()
    try:
        output = args.output or args.data_root / "binary_cifar100"
        build_cifar100_cache(
            args.data_root,
            output,
            download=args.download,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cifar100_data_cli())
