import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cl_datasets import (
    BinarySplitCIFAR100,
    build_torchvision_dataset,
    ensure_cifar100_cache,
)


class SNNDataManager:
    def __init__(self, args):
        self.dataset = args.dataset
        self.batch_train = args.batch_size
        self.batch_test = args.val_batch_size
        self.seed = args.seed
        self.num_tasks = args.num_tasks
        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.data_path = os.path.expanduser(args.data_path)
        self.n_steps = args.n_steps
        self.tasks_order = np.arange(
            self.num_tasks * self.num_classes_per_task
        ).reshape(self.num_tasks, self.num_classes_per_task)
        self.train_dataset = self._build_dataset(True)
        self.test_dataset = self._build_dataset(False)
        args.full_train_len = len(self.train_dataset)
        args.full_test_len = len(self.test_dataset)
        args.n_classes = self.num_classes

    def _build_dataset(self, train):
        if self.dataset == "cifar100":
            return BinarySplitCIFAR100(
                ensure_cifar100_cache(self.data_path),
                train=train,
            )

        import torchvision.transforms as transforms

        if self.dataset == "mnist":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda image: image.repeat(3, 1, 1)),
                    transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
                ]
            )
        elif self.dataset == "fashionmnist":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda image: image.repeat(3, 1, 1)),
                    transforms.Normalize((0.286,) * 3, (0.353,) * 3),
                ]
            )
        elif self.dataset == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.201),
                    ),
                ]
            )
        else:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        return build_torchvision_dataset(
            self.dataset,
            self.data_path,
            train,
            transform,
            self.tasks_order,
        )

    def get_loader(self, dataset, task, mode="train"):
        dataset.set_task(task)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(task))
        return DataLoader(
            dataset,
            batch_size=self.batch_train if mode == "train" else self.batch_test,
            collate_fn=self._temporal_collate_fn,
            generator=generator,
        )

    def _temporal_collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return (
            images.unsqueeze(0).expand(self.n_steps, -1, -1, -1, -1),
            labels,
        )
