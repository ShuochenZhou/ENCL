import os

import torch
import torch.nn.functional as F
from .local_base import LocalLearningLearner
from ..common.experiment import print_and_log


class TESSNoNCLLearner(LocalLearningLearner):
    def __init__(self, args, data_manager):
        self._initialize_common(args, data_manager)

    def _get_model(self, num_classes):
        self.args.n_classes = num_classes
        self.args.num_classes = num_classes
        from ..models.tess import vgg_tess
        model = vgg_tess(self.args, self.device)
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
        self.args.lr_conv = float(self.args.lr_conv)
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
        current_head_parameters = self._current_head_update_parameters()
        for epoch in range(self.epochs):
            train_loss = 0
            correct = 0
            n = 0
            self._reset_local_learning_statistics()
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
                loss_accumulated = 0
                self._clear_current_head_gradients(current_head_parameters)
                self.net.clear_local_gradients()
                time_steps = temporal_data.size(0)
                ce_delay_eff = int(time_steps)
                ce_delay_override = int(time_steps)
                signal_steps = 0
                loss_accumulated = 0.0
                timestep_stats = []
                for t in range(time_steps):
                    input_t = temporal_data[t] if temporal_data.size(0) > 1 else temporal_data[0]
                    remaining_steps = time_steps - t
                    target_for_local = labels
                    features_t = self.net.forward_features(input_t, target_for_local)
                    output = self._mr_head_full_logits(features_t, class_start, class_end)
                    if pred is None:
                        pred = output.detach()
                    else:
                        pred = pred + output.detach()
                    if remaining_steps <= ce_delay_eff:
                        relevant_output = output[:, class_start:class_end]
                        if batch_idx == 0 and epoch == 0 and (t == time_steps - 1):
                            with torch.no_grad():
                                mean_logits = relevant_output.mean(dim=0).cpu().tolist()
                                if relevant_output.size(0) >= 3:
                                    sample_logits = relevant_output[:3].cpu().tolist()
                                pred_classes_local = relevant_output.argmax(dim=1)
                                class_counts = {}
                                for cls in range(relevant_output.size(1)):
                                    class_counts[cls] = (pred_classes_local == cls).sum().item()
                        if relevant_output.size(1) == 1:
                            step_loss = F.binary_cross_entropy_with_logits(relevant_output.squeeze(1), dummy_targets.float())
                        else:
                            step_loss = F.cross_entropy(relevant_output, dummy_targets, weight=None)
                        step_loss.backward()
                        if batch_idx == 0:
                            try:
                                conv_grad = None
                                for (name, p) in self.net.named_parameters():
                                    if p.grad is None:
                                        continue
                                    if 'conv' in name or p.dim() >= 4:
                                        conv_grad = float(p.grad.norm().item())
                                        break
                            except Exception:
                                pass
                        loss_accumulated += step_loss.item()
                        signal_steps += 1
                if signal_steps > 0:
                    loss = torch.tensor(loss_accumulated / signal_steps, device=self.device)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                self._update_current_head(current_head_parameters)
                self.net.apply_local_plain_sgd()
                log_pred = pred.clone().detach() if pred is not None else None
                self.net.eval()
                with torch.no_grad():
                    if log_pred is None:
                        print_and_log('ERROR: pred is None, cannot compute logging loss!')
                        loss = torch.tensor(0.0, device=self.device)
                        relevant_logits = None
                    else:
                        relevant_logits = log_pred[:, class_start:class_end]
                    if log_pred is not None and (torch.isnan(log_pred).any() or torch.isinf(log_pred).any()):
                        if batch_idx == 0:
                            print_and_log(f'WARNING: NaN or Inf in logits!')
                        loss = torch.tensor(0.0, device=self.device)
                    elif relevant_logits is not None:
                        if batch_idx == 0:
                            try:
                                rel_min = float(relevant_logits.min().item())
                                rel_max = float(relevant_logits.max().item())
                                rel_mean = float(relevant_logits.mean().item())
                                rel_std = float(relevant_logits.std().item())
                                (unique_targets, counts) = torch.unique(dummy_targets, return_counts=True)
                                target_hist = {int(k.item()): int(v.item()) for (k, v) in zip(unique_targets, counts)}
                            except Exception:
                                pass
                        if relevant_logits.size(1) == 1:
                            loss = F.binary_cross_entropy_with_logits(relevant_logits.squeeze(1), dummy_targets.float())
                        elif (dummy_targets < 0).any() or (dummy_targets >= relevant_logits.size(1)).any():
                            print_and_log(f'ERROR: Invalid target indices at batch {batch_idx}! dummy_targets min={dummy_targets.min()}, max={dummy_targets.max()}, num_classes={relevant_logits.size(1)}')
                            print_and_log(f'ERROR: class_start={class_start}, num_classes_per_task={self.data_manager.num_classes_per_task}')
                            print_and_log(f'ERROR: This means loss will be 0. Please check class indexing!')
                            loss = torch.tensor(0.0, device=self.device)
                        else:
                            loss = F.cross_entropy(relevant_logits, dummy_targets, weight=None)
                        if loss.item() < 1e-06:
                            if batch_idx == 0:
                                print_and_log(f'WARNING: Loss is very small ({loss.item():.8f}) - might indicate problem with predictions!')
                        if torch.isnan(loss) or torch.isinf(loss):
                            print_and_log(f'WARNING: Loss is NaN or Inf! Setting to 0.')
                            loss = torch.tensor(0.0, device=self.device)
                self.net.train()
                if log_pred is not None:
                    pred = log_pred.detach()
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
            self._log_local_learning_statistics(epoch)
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
