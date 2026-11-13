import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.experiment import get_experiment_base_dir, print_and_log


class BaseContinualLearner:
    def _initialize_common(self, args, data_manager):
        self.cur_task = 0
        self._known_classes = 0
        self.data_manager = data_manager
        self.args = args
        self.exp_folder = args.experiment_folder
        exp_base_dir = get_experiment_base_dir(args)
        self.models_dir = f'{exp_base_dir}/{self.exp_folder}/models'
        self.masks_dir = f'{exp_base_dir}/{self.exp_folder}/masks'
        self.exp_dir = f'{exp_base_dir}/{self.exp_folder}'
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.accuracy_log_file = f'{self.exp_dir}/accuracy_log.txt'
        self.memory_log_file = f'{self.exp_dir}/memory_log.txt'
        if torch.cuda.is_available() and not getattr(args, 'cpu', False):
            n_gpus = torch.cuda.device_count()
            requested = args.gpu_index
            if 0 <= requested < n_gpus:
                gpu_index = requested
            else:
                free_mem = []
                for index in range(n_gpus):
                    try:
                        free, _ = torch.cuda.mem_get_info(index)
                    except Exception:
                        free = 0
                    free_mem.append(free)
                gpu_index = int(max(range(n_gpus), key=lambda index: free_mem[index]))
                print_and_log(f'Auto-selected GPU {gpu_index} (free MB per GPU: {[round(f / 1024 ** 2) for f in free_mem]})')
            self.device = torch.device(f'cuda:{gpu_index}')
            torch.cuda.set_device(gpu_index)
            self.gpu_index = gpu_index
            print_and_log(f'Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}')
        else:
            self.device = torch.device('cpu')
            self.gpu_index = None
            print_and_log('Using CPU')
        self._init_log_files()
        self.epochs = args.epochs
        self.n_steps = args.n_steps
        self.task_heads = {}
        self.mr_task_heads = None
        self._best_mr_head_state = None
        self.net = None
        self.NET = None
        self.best_acc = -1.0
        self.acc_matrix = np.zeros(
            (self.data_manager.num_tasks, self.data_manager.num_tasks)
        )
        self.bwt = np.zeros(self.data_manager.num_tasks)
        self.memory_entries = []
    def _get_task_class_range(self, task_idx):
        total_classes = self.data_manager.num_classes
        per_task = self.data_manager.num_classes_per_task
        class_start = task_idx * per_task
        class_end = min(class_start + per_task, total_classes)
        return (class_start, class_end)

    def _ensure_mr_task_heads(self, feature_dim, classes_per_task):
        feature_dim = int(feature_dim)
        classes_per_task = int(classes_per_task)
        if self.mr_task_heads is None:
            self.mr_task_heads = nn.ModuleList([nn.Linear(feature_dim, classes_per_task) for _ in range(int(self.data_manager.num_tasks))]).to(self.device)
        head = self.mr_task_heads[int(self.cur_task)]
        return head

    def _current_mr_task_head(self):
        if self.mr_task_heads is None:
            return None
        return self.mr_task_heads[int(self.cur_task)]

    def _mr_head_full_logits(self, features, class_start, class_end):
        head = self._current_mr_task_head()
        local_logits = head(features)
        total_classes = self.data_manager.num_classes
        return F.pad(local_logits, (int(class_start), total_classes - int(class_end)))

    def _load_compatible_state_dict(self, model, state_dict):
        model.load_state_dict(state_dict, strict=False)
        return

    def _init_log_files(self):
        with open(self.accuracy_log_file, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('Accuracy Log - Per Task Evaluation\n')
            f.write('=' * 80 + '\n')
            f.write(f'Dataset: {self.data_manager.dataset}\n')
            f.write(f'Number of Tasks: {self.data_manager.num_tasks}\n')
            f.write(f'Number of Classes: {self.data_manager.num_classes}\n')
            f.write(f'Classes per Task: {self.data_manager.num_classes_per_task}\n')
            f.write('=' * 80 + '\n\n')
        with open(self.memory_log_file, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('GPU Memory Usage Log\n')
            f.write('=' * 80 + '\n')
            f.write(f'GPU Index: {self.gpu_index}\n')
            if torch.cuda.is_available():
                f.write(f'GPU Name: {torch.cuda.get_device_name(self.gpu_index)}\n')
            f.write('=' * 80 + '\n\n')
            f.write(f"{'Timestamp':<20} {'Event':<30} {'Memory Allocated (MB)':<25} {'Memory Reserved (MB)':<25}\n")
            f.write('-' * 80 + '\n')

    def _log_memory_usage(self, event_name):
        if self.gpu_index is not None and torch.cuda.is_available():
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            memory_allocated = torch.cuda.memory_allocated(self.gpu_index) / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved(self.gpu_index) / 1024 ** 2
            self.memory_entries.append({'timestamp': timestamp, 'event': event_name, 'allocated_mb': memory_allocated, 'reserved_mb': memory_reserved})
            with open(self.memory_log_file, 'a') as f:
                f.write(f'{timestamp:<20} {event_name:<30} {memory_allocated:<25.2f} {memory_reserved:<25.2f}\n')
            print_and_log(f'Memory [{event_name}]: Allocated={memory_allocated:.2f} MB, Reserved={memory_reserved:.2f} MB')
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(self.memory_log_file, 'a') as f:
                f.write(f"{timestamp:<20} {event_name:<30} {'N/A (CPU)':<25} {'N/A (CPU)':<25}\n")

    def _log_task_accuracy(self, task_id):
        with open(self.accuracy_log_file, 'a') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f'After Training Task {task_id}\n')
            f.write(f"{'=' * 80}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
            dense_path = f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt'
            if os.path.exists(dense_path):
                if self.NET is None:
                    self.NET = self._get_model(self.data_manager.num_classes)
                    self._load_compatible_state_dict(self.NET, torch.load(dense_path, map_location=self.device))
                    self.NET = self.NET.to(self.device)
                f.write('Accuracy Matrix (rows = evaluation time, cols = task):\n')
                f.write('-' * 80 + '\n')
                f.write(f'\nEvaluating all tasks learned so far (tasks 0 to {task_id}):\n')
                task_accuracies = []
                for t in range(task_id + 1):
                    test_loader = self.data_manager.get_loader(self.data_manager.test_dataset, t, mode='test')
                    selected_mask = t
                    self.net = self._set_mask(selected_mask)
                    self.net = self.net.to(self.device)
                    classes_per_task = getattr(self.data_manager, 'num_classes_per_task', None)
                    if classes_per_task is None or classes_per_task == 0:
                        classes_per_task = max(1, self.data_manager.num_classes // self.data_manager.num_tasks)
                    class_start = t * classes_per_task
                    class_end = min((t + 1) * classes_per_task, self.data_manager.num_classes)
                    test_accuracy = self._compute_accuracy(test_loader, class_range=(class_start, class_end))
                    task_accuracies.append(test_accuracy)
                    self.acc_matrix[task_id, t] = test_accuracy
                    f.write(f'\n  Task {t} (Classes {class_start}-{class_end - 1}):\n')
                    f.write(f'    Task Accuracy: {test_accuracy:.2f}% (Expert {selected_mask})\n')
                avg_acc = np.mean(task_accuracies) if task_accuracies else 0.0
                f.write(f'\n  Summary:\n')
                f.write(f'    Average Task Accuracy (0-{task_id}): {avg_acc:.2f}%\n')
                f.write('\n')
            else:
                f.write('WARNING: Dense model not found. Cannot evaluate accuracy.\n')
            f.write(f"{'=' * 80}\n\n")

    def after_task(self):

        self._log_memory_usage(f'Task {self.cur_task} - After Training, Before Post-Processing')
        checkpoint_path = f'{self.models_dir}/{self.data_manager.dataset}.pt'
        if os.path.exists(checkpoint_path):
            self.net.eval()
            self._load_compatible_state_dict(self.net, torch.load(checkpoint_path, map_location=self.device))
            self.net = self.net.to(self.device)
        self._stack_subnets()
        self._train_per_task_head()
        self._log_memory_usage(f'Task {self.cur_task} - After Stacking Subnetworks')
        if getattr(self, '_meter', None) is not None:
            self._meter.mark_train_end(self.cur_task)
        dense_path = f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt'
        if os.path.exists(dense_path):
            self._log_task_accuracy(self.cur_task)
            self.net = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print_and_log(f'WARNING: Dense model not found after task {self.cur_task}. Skipping accuracy logging.')
        self._log_memory_usage(f'Task {self.cur_task} - After Accuracy Evaluation')
        self.cur_task += 1
        self.best_acc = -1.0
        self._known_classes += self.data_manager.num_classes_per_task

    def _stack_subnets(self):
        was_training = self.net.training
        self.net.eval()
        self.NET = self._get_model(self.data_manager.num_classes)
        self._load_compatible_state_dict(self.NET, self.net.state_dict())
        self.NET = self.NET.to(self.device)
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for (k, v) in self.NET.state_dict().items()}, f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt')
        if was_training:
            self.net.train()
        return

    def _set_mask(self, task_id):
        if self.NET is None:
            raise ValueError('Dense noNCL resident model is not available.')
        dense_net = self._get_model(self.data_manager.num_classes)
        self._load_compatible_state_dict(dense_net, self.NET.state_dict())
        dense_net = dense_net.to(self.device)
        return dense_net

    def _save_checkpoint(self):
        os.makedirs(self.models_dir, exist_ok=True)
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for (k, v) in self.net.state_dict().items()}
        torch.save(state_dict, f'{self.models_dir}/{self.data_manager.dataset}.pt')

    def _train_per_task_head(self):
        head = self._current_mr_task_head()
        self.task_heads[int(self.cur_task)] = {'kind': 'featurehead', 'w': head.weight.detach().cpu(), 'b': head.bias.detach().cpu(), 'feature_dim': int(head.in_features)}
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(self.task_heads, f'{self.models_dir}/{self.data_manager.dataset}_feature_heads.pt')
        print_and_log(f'NO HEAD-ONLY PHASE task={self.cur_task}: stored the main-phase Linear({head.in_features},{head.out_features}) head unchanged.')

    def _featurehead_accuracy(self, iterator, class_range):
        (cs, ce) = class_range
        task_idx = int(cs // self.data_manager.num_classes_per_task)
        head = self.task_heads[task_idx]
        w = head['w'].to(self.device)
        b = head['b'].to(self.device)
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for (data, labels) in iterator:
                data = data.to(self.device)
                labels = labels.type(torch.int64).to(self.device)
                self.net.reset_states()
                feature_sum = None
                for t in range(self.n_steps):
                    features = self.net.forward_features(data[t], None)
                    feature_sum = features if feature_sum is None else feature_sum + features
                predictions = (feature_sum @ w.t() + b).argmax(1) + cs
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
        return 100.0 * correct / total if total else 0.0

    def _compute_accuracy(self, loader, class_range):
        return self._featurehead_accuracy(loader, class_range)

    def evaluate(self):

        print_and_log('\nTesting...')
        if self.cur_task == 0:
            print_and_log('No tasks available for evaluation yet.')
            return
        self._log_memory_usage('Final Evaluation - Start')
        dense_path = f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt'
        if os.path.exists(dense_path):
            self.NET = self._get_model(self.data_manager.num_classes)
            self._load_compatible_state_dict(self.NET, torch.load(dense_path, map_location=self.device))
            self.NET = self.NET.to(self.device)
            self._log_memory_usage('Final Evaluation - After Loading Dense Model')
        else:
            raise FileNotFoundError(f'Dense model not found: {dense_path}')
        for task in range(self.cur_task - 1, self.cur_task):
            for t in range(task + 1):
                self.test_loader = self.data_manager.get_loader(self.data_manager.test_dataset, t, mode='test')
                selected_mask = t
                (class_start, class_end) = self._get_task_class_range(t)
                self.net = self._set_mask(selected_mask)
                self.net = self.net.to(self.device)
                test_accuracy = self._compute_accuracy(self.test_loader, class_range=(class_start, class_end))
                self.acc_matrix[task, t] = test_accuracy
                print_and_log(f'Task {task}, Test Task {t}: Expert {selected_mask}, Accuracy: {test_accuracy:.2f}%')
        self._log_memory_usage('Final Evaluation - Complete')

    def report_scores(self):

        print_and_log('\nACCURACY MATRIX:\n {}'.format(self.acc_matrix))
        for task in range(self.cur_task - 1):
            if task < len(self.bwt):
                self.bwt[task] = self.acc_matrix[self.cur_task - 1, task] - self.acc_matrix[task, task]
        print_and_log('\nBWT (Backward Transfer):\n {}'.format(self.bwt))
        final_acc = self.acc_matrix[-1]
        inc_acc = np.zeros_like(final_acc)
        for i in range(len(final_acc)):
            if i == 0:
                inc_acc[i] = final_acc[i]
            else:
                inc_acc[i] = np.mean(final_acc[:i + 1])
        print_and_log('\nINCREMENTAL ACCURACY:\n {}'.format(inc_acc))
        if len(self.bwt) > 0:
            avg_bwt = np.mean([b for b in self.bwt if not np.isnan(b)])
            print_and_log('\nAverage BWT: {:.4f}'.format(avg_bwt))
