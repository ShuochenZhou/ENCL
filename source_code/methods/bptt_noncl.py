import os

import torch
import torch.nn.functional as F

from .base import BaseContinualLearner
from ..common.experiment import (
    clear_gradients,
    plain_sgd_update,
    print_and_log,
)


class BPTTNoNCLLearner(BaseContinualLearner):
    def __init__(self, args, data_manager):
        self._initialize_common(args, data_manager)

    def _get_model(self, num_classes):
        self.args.n_classes = num_classes
        self.args.num_classes = num_classes
        from ..models.bptt import vgg_bptt
        model = vgg_bptt(self.args, self.device)
        if self.cur_task == 0 and (not hasattr(self, '_model_output_checked')):
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                try:
                    if hasattr(model, 'forward'):
                        pass
                except Exception as error:
                    print_and_log(f'Note: Could not verify model output shape: {error}')
            self._model_output_checked = True
        return model

    def train(self):

        print_and_log('\nTraining...')
        if getattr(self, '_meter', None) is not None:
            self._meter.mark_train_start(self.cur_task)
        self._log_memory_usage(f'Task {self.cur_task} - Start Training (Trial 0)')
        self.net = self._get_model(self.data_manager.num_classes)
        self.net = self.net.to(self.device)
        self._log_memory_usage(f'Task {self.cur_task} - After Model Creation')
        self.train_loader = self.data_manager.get_loader(self.data_manager.train_dataset, self.cur_task, mode='train')
        (class_start, class_end) = self._get_task_class_range(self.cur_task)
        if self.cur_task != 0:
            dense_path = f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt'
            if os.path.exists(dense_path):
                state_dict = torch.load(dense_path, map_location=self.device)
                self._load_compatible_state_dict(self.net, state_dict)
        feature_dim = getattr(self.net, 'feature_dim', None)
        current_head = self._ensure_mr_task_heads(feature_dim, class_end - class_start)
        self._best_mr_head_state = None
        update_groups = self._build_update_groups(self.args.lr_conv)
        for epoch in range(self.epochs):
            train_loss = 0
            correct = 0
            n = 0
            for (batch_idx, (data, labels)) in enumerate(self.train_loader):
                if isinstance(data, tuple) or isinstance(data, list):
                    if len(data) == 2:
                        (data, labels) = data
                labels = labels.type(torch.int64)
                (data, labels) = (data.to(self.device), labels.to(self.device))
                if data.dim() == 5 and data.size(0) == self.n_steps:
                    temporal_data = data
                elif data.dim() == 5 and data.size(1) == self.n_steps:
                    temporal_data = data.permute(1, 0, 2, 3, 4)
                elif data.dim() == 4:
                    temporal_data = data.unsqueeze(0).expand(self.n_steps, -1, -1, -1, -1)
                else:
                    temporal_data = data
                self.net.reset_states()
                self.net.train()
                classes_per_task = getattr(self.data_manager, 'num_classes_per_task', None)
                if classes_per_task is None or classes_per_task <= 0:
                    dummy_targets = labels.clone()
                else:
                    dummy_targets = labels - class_start
                if classes_per_task is not None and classes_per_task > 0 and (dummy_targets < 0).any():
                    invalid_mask = dummy_targets < 0
                    invalid_labels = labels[invalid_mask]
                    print_and_log(f'ERROR: Found labels less than class_start! Labels: {invalid_labels.tolist()}, class_start: {class_start}')
                    print_and_log(f'ERROR: This indicates labels from previous tasks are in current task data!')
                    dummy_targets = torch.clamp(dummy_targets, min=0)
                if classes_per_task is not None and classes_per_task > 0 and (dummy_targets >= classes_per_task).any():
                    invalid_mask = dummy_targets >= self.data_manager.num_classes_per_task
                    invalid_labels = labels[invalid_mask]
                    invalid_dummy = dummy_targets[invalid_mask]
                    print_and_log(f'ERROR: Found labels beyond current task range! Labels: {invalid_labels.tolist()}, dummy_targets: {invalid_dummy.tolist()}')
                    print_and_log(f'ERROR: num_classes_per_task: {self.data_manager.num_classes_per_task}')
                    dummy_targets = torch.clamp(dummy_targets, max=self.data_manager.num_classes_per_task - 1)
                pred = None
                for group in update_groups:
                    clear_gradients(group['parameters'])
                for t in range(self.n_steps):
                    input_t = temporal_data[t] if temporal_data.size(0) > 1 else temporal_data[0]
                    features_t = self.net.forward_features(input_t, None)
                    output = self._mr_head_full_logits(features_t, class_start, class_end)
                    if pred is None:
                        pred = output
                    else:
                        pred = pred + output
                relevant_pred = pred[:, class_start:class_end]
                if relevant_pred.size(1) == 1:
                    loss = F.binary_cross_entropy_with_logits(relevant_pred.squeeze(1), dummy_targets.float())
                else:
                    loss = F.cross_entropy(relevant_pred, dummy_targets, weight=None)
                loss.backward()
                for group in update_groups:
                    plain_sgd_update(group['parameters'], group['lr'])
                if pred is not None:
                    pred = pred.detach()
                train_loss += loss.item()
                with torch.no_grad():
                    if pred is not None:
                        if class_end > class_start:
                            task_logits = pred[:, class_start:class_end]
                            preds_local = task_logits.argmax(dim=1, keepdim=True)
                            preds = preds_local + class_start
                        else:
                            preds = pred.argmax(dim=1, keepdim=True)
                        correct += preds.eq(labels.view_as(preds)).sum().item()
                        n += len(labels)
                if batch_idx % 50 == 0:
                    batch_acc = 100 * correct / n if n > 0 else 0
                    print_and_log('Batch {:3d} | Train: acc={:5.2f}% |'.format(batch_idx, batch_acc))
            train_acc = 100 * correct / n if n > 0 else 0
            avg_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else train_loss
            print_and_log('\nSummary => Train: loss={:.3f}, acc={:5.2f}% |'.format(avg_loss, train_acc))
            if train_acc > self.best_acc:
                self.best_acc = train_acc
                current_head = self._current_mr_task_head()
                self._best_mr_head_state = {name: value.detach().clone() for (name, value) in current_head.state_dict().items()}
                self.net.eval()
                self._save_checkpoint()
                self.net.train()
        if self._best_mr_head_state is not None:
            self._current_mr_task_head().load_state_dict(self._best_mr_head_state)
        return train_acc

    def _build_backbone_update_groups(self, base_lr, lr_conv):
        conv_params = []
        other_params = []
        for (name, param) in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if 'hlop' in name.lower():
                continue
            name_lower = name.lower()
            is_conv_like = 'conv' in name_lower
            if is_conv_like:
                conv_params.append(param)
            else:
                other_params.append(param)
        update_groups = []
        if conv_params:
            update_groups.append({'parameters': conv_params, 'lr': lr_conv})
        if other_params:
            update_groups.append({'parameters': other_params, 'lr': base_lr})
        return update_groups

    def _build_update_groups(self, lr):
        lr_conv = getattr(self.args, 'lr_conv', lr)
        update_groups = self._build_backbone_update_groups(
            base_lr=lr,
            lr_conv=lr_conv,
        )
        current_head = self._current_mr_task_head()
        task_head_lr = float(self.args.feature_head_lr)
        update_groups.append({
            'parameters': list(current_head.parameters()),
            'lr': task_head_lr,
        })
        return update_groups
