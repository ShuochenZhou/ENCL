import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.experiment import (
    clear_gradients,
    count_bytes,
    local_labels,
    plain_sgd_update,
    seed_all,
)
from ..common.resource_meter import ResourceMeter
from ..data.snn_datamanager import SNNDataManager
from ..models.bptt import VGG
from ..models.bptt_layers.surrogate_gradients import GradSigmoid


class SharedVGG9(nn.Module):
    channels = (64, 128, 256, 256, 512, 512, 512, 512)

    def __init__(self, tasks, classes_per_task, steps):
        super().__init__()
        self.steps = int(steps)
        self.core = VGG(
            3,
            classes_per_task,
            GradSigmoid,
            dropout=0.0,
            feature_only=True,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(512, classes_per_task) for _ in range(tasks)]
        )

    def one(self, x):
        core = self.core
        x = core.dropout1(core.lif1(core.conv1(x)))
        x = core.pool1(core.dropout2(core.lif2(core.conv2(x))))
        x = core.dropout3(core.lif3(core.conv3(x)))
        x = core.pool2(core.dropout4(core.lif4(core.conv4(x))))
        x = core.dropout5(core.lif8(core.conv8(x)))
        x = core.pool4(core.dropout6(core.lif9(core.conv9(x))))
        x = core.dropout7(core.lif11(core.conv11(x)))
        x = core.dropout8(core.lif12(core.conv12(x)))
        return core.globalpool(x).flatten(1)

    def forward(self, x, task):
        self.core.reset_states()
        frames = (
            [x[index] for index in range(self.steps)]
            if x.dim() == 5
            else [x] * self.steps
        )
        logits = None
        for frame in frames:
            step_logits = self.heads[task](self.one(frame))
            logits = step_logits if logits is None else logits + step_logits
        return logits


def build_data_manager(args):
    return SNNDataManager(
        SimpleNamespace(
            dataset=args.dataset,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            seed=args.seed,
            num_tasks=args.num_tasks,
            num_classes=args.num_classes,
            num_classes_per_task=args.num_classes_per_task,
            data_path=args.data_path,
            n_steps=args.n_steps,
        )
    )


@torch.no_grad()
def evaluate(model, loader, task, classes_per_task, device):
    model.eval()
    correct = 0
    count = 0
    for (images, labels) in loader:
        predictions = model(images.to(device), task).argmax(1)
        labels = local_labels(labels.to(device), task, classes_per_task)
        correct += (predictions == labels).sum().item()
        count += len(labels)
    return 100 * correct / count


def run_experiment(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meter = ResourceMeter(
        args.measure_gpu_index,
        args.out,
        idle_seconds=args.idle_seconds,
    )
    seed_all(args.seed)
    device = torch.device(f'cuda:{args.gpu_index}')
    meter.mark_train_start(0)
    model = SharedVGG9(
        args.num_tasks,
        args.num_classes_per_task,
        args.n_steps,
    ).to(device)
    data = build_data_manager(args)
    accuracy = np.zeros((args.num_tasks, args.num_tasks))
    replay_images = None
    replay_labels = None
    replay_tasks = None
    seen = 0
    for task in range(args.num_tasks):
        if task > 0:
            meter.mark_train_start(task)
        train_loader = data.get_loader(data.train_dataset, task, mode='train')
        parameters = list(model.core.parameters())
        for head in model.heads[:task + 1]:
            parameters.extend(head.parameters())
        best_accuracy = -1.0
        best_epoch = -1
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            correct = 0
            count = 0
            for (images, labels) in train_loader:
                images = images.to(device, non_blocking=True)
                labels = local_labels(
                    labels.to(device, non_blocking=True),
                    task,
                    args.num_classes_per_task,
                )
                clear_gradients(parameters)
                current_logits = model(images, task)
                loss = F.cross_entropy(current_logits, labels)
                correct += (current_logits.detach().argmax(1) == labels).sum().item()
                count += labels.numel()
                if replay_images is not None:
                    sample = torch.randint(len(replay_labels), (min(args.replay_batch, len(replay_labels)),))
                    represented_tasks = replay_tasks[sample].unique().tolist()
                    replay_loss = 0
                    for old_task in represented_tasks:
                        selected = replay_tasks[sample] == old_task
                        replay_loss = replay_loss + F.cross_entropy(model(replay_images[sample][selected].to(device), int(old_task)), replay_labels[sample][selected].to(device))
                    loss = loss + replay_loss / len(represented_tasks)
                loss.backward()
                plain_sgd_update(parameters, args.lr)
            train_accuracy = 100.0 * correct / count
            selected = train_accuracy > best_accuracy
            if selected:
                best_accuracy = float(train_accuracy)
                best_epoch = epoch + 1
                best_state = {name: value.detach().cpu().clone() for (name, value) in model.state_dict().items()}
            print(f"[{time.strftime('%F %T')}] mr task={task} epoch={epoch + 1}/{args.epochs} train_acc={train_accuracy:.6f}% best_train_acc={best_accuracy:.6f}% best_epoch={best_epoch} selected={selected}", flush=True)
        model.load_state_dict(best_state, strict=True)
        del best_state
        data.train_dataset.set_task(task)
        for index in range(len(data.train_dataset)):
            (image, global_label) = data.train_dataset[index]
            label = torch.tensor(
                int(global_label) - args.num_classes_per_task * task,
                dtype=torch.long,
            )
            seen += 1
            if replay_images is None:
                replay_images = image.unsqueeze(0)
                replay_labels = label.view(1)
                replay_tasks = torch.tensor([task])
            elif len(replay_labels) < args.replay_size:
                replay_images = torch.cat([replay_images, image.unsqueeze(0)])
                replay_labels = torch.cat([replay_labels, label.view(1)])
                replay_tasks = torch.cat([replay_tasks, torch.tensor([task])])
            else:
                replacement = random.randrange(seen)
                if replacement < args.replay_size:
                    replay_images[replacement] = image
                    replay_labels[replacement] = label
                    replay_tasks[replacement] = task
        meter.mark_train_end(task)
        for old_task in range(task + 1):
            test_loader = data.get_loader(data.test_dataset, old_task, mode='test')
            accuracy[task, old_task] = evaluate(
                model,
                test_loader,
                old_task,
                args.num_classes_per_task,
                device,
            )
        print(f"[{time.strftime('%F %T')}] mr row={accuracy[task, :task + 1].tolist()} avg={accuracy[task, :task + 1].mean():.3f}", flush=True)
    diagonal = np.diag(accuracy)
    curve = [
        float(accuracy[index, :index + 1].mean())
        for index in range(args.num_tasks)
    ]
    bwt_curve = [
        0.0
        if index == 0
        else float(np.mean(accuracy[index, :index] - diagonal[:index]))
        for index in range(args.num_tasks)
    ]
    model_bytes = sum((parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    replay_bytes = sum((count_bytes(value) for value in (replay_images, replay_labels, replay_tasks)))
    result = meter.finalize(sum((parameter.numel() for parameter in model.parameters())), model_bytes / 1000000.0, {'method': 'MR', 'dataset': args.dataset, 'final_avg_accuracy_pct': float(accuracy[-1].mean()), 'avg_acc_curve': curve, 'avg_bwt_pct': bwt_curve[-1], 'avg_bwt_curve': bwt_curve, 'acc_matrix': accuracy.tolist(), 'method_aux_bytes': replay_bytes, 'memory_breakdown_bytes': {'model': model_bytes, 'method_state': replay_bytes}})
    print('[final] ' + json.dumps(result), flush=True)
