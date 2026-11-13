import copy
import math
import os

import torch
import torch.nn.functional as F
from .local_base import LocalLearningLearner
from .masking import Masking
from ..common.experiment import print_and_log


# ENCL extends the standard TESS procedure with task-wise sparse allocation and isolation.
class ENCLLearner(LocalLearningLearner):
    def __init__(self, args, data_manager):
        self.used_params = {}
        self._initialize_common(args, data_manager)
        self.isolate = args.isolate
        self.refill_overlap = args.refill_overlap
        self.max_overlap_fraction = args.max_overlap_fraction
        self.sparse_init = args.sparse_init
        self.current_task_mask = {}

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

    def _compute_fan_in(self, param):
        if param.dim() == 2:
            return param.shape[1]
        if param.dim() > 2:
            fan_in = param.shape[1]
            for dim in param.shape[2:]:
                fan_in *= dim
            return fan_in
        return max(param.numel(), 1)

    def _reset_new_weights(self, name, new_mask_only):
        # ENCL initialization: reset only connections newly allocated to the current task.
        if new_mask_only is None or not isinstance(new_mask_only, torch.Tensor):
            return
        params = dict(self.net.named_parameters()) if self.net is not None else {}
        target_param = params.get(name)
        if target_param is None:
            return
        mask_bool = new_mask_only.to(target_param.device).bool()
        if not mask_bool.any():
            return
        fan_in = self._compute_fan_in(target_param)
        bound = 1.0 / math.sqrt(max(fan_in, 1))
        with torch.no_grad():
            n_fresh = int(mask_bool.sum().item())
            rand_vals = torch.empty(n_fresh, device=target_param.device, dtype=target_param.dtype).uniform_(-bound, bound)
            target_param.data[mask_bool] = rand_vals

    def after_task(self):

        self._log_memory_usage(f'Task {self.cur_task} - After Training, Before Post-Processing')
        checkpoint_path = f'{self.models_dir}/{self.data_manager.dataset}.pt'
        if os.path.exists(checkpoint_path):
            self.net.eval()
            self._load_compatible_state_dict(self.net, torch.load(checkpoint_path, map_location=self.device))
            self.net = self.net.to(self.device)
        self._update_used_params()
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
        self.mask = Masking(
            net=self.net,
            density=float(self.args.mask_density),
            sparse_init=self.sparse_init,
            device=self.device,
        )
        self._prepare_masks_for_task()
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
                self._freeze_used_params()
                self._update_current_head(current_head_parameters)
                self.net.apply_local_plain_sgd()
                self.mask.step(epoch)
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
                self._save_mask()
                self.net.train()
        if self._best_mr_head_state is not None:
            self._current_mr_task_head().load_state_dict(self._best_mr_head_state)
        return train_acc

    def _prepare_masks_for_task(self):
        # ENCL allocation: bound reuse of old connections and refill the mask with free ones.
        self.current_task_mask = {}
        prev_masks = self.used_params if self.used_params else {}
        for (name, mask_tensor) in list(self.mask.masks.items()):
            if 'normalization' in name:
                ones_mask = torch.ones_like(mask_tensor, dtype=torch.float32, device=self.device)
                self.mask.masks[name] = ones_mask
                continue
            mask_bool = mask_tensor.to(self.device).bool()
            prev_active = prev_masks.get(name)
            if isinstance(prev_active, torch.Tensor):
                prev_bool = prev_active.to(self.device).bool()
            else:
                prev_bool = torch.zeros_like(mask_bool, dtype=torch.bool, device=self.device)
            desired_mask = mask_bool.clone()
            available_mask = (~prev_bool).clone()
            if self.refill_overlap:
                target_nnz = int(desired_mask.sum().item())
                overlap = desired_mask & prev_bool
                overlap_idx = torch.where(overlap.flatten())[0]
                target_overlap = min(int(prev_bool.sum().item()), int(round(target_nnz * self.max_overlap_fraction)))
                keep_overlap = min(int(overlap_idx.numel()), target_overlap)
                if overlap_idx.numel() > keep_overlap:
                    perm = torch.randperm(overlap_idx.numel(), device=self.device)
                    drop_idx = overlap_idx[perm[keep_overlap:]]
                    compensated = desired_mask.flatten().clone()
                    compensated[drop_idx] = False
                    free = (~prev_bool & ~compensated.reshape_as(prev_bool)).flatten()
                    free_idx = torch.where(free)[0]
                    need = target_nnz - int(compensated.sum().item())
                    if need > 0 and free_idx.numel() > 0:
                        take = min(need, int(free_idx.numel()))
                        choice = free_idx[torch.randperm(free_idx.numel(), device=self.device)[:take]]
                        compensated[choice] = True
                    desired_mask = compensated.reshape_as(desired_mask)
                    available_mask = (~prev_bool).clone()
            new_mask_only = desired_mask & available_mask
            self.current_task_mask[name] = new_mask_only.float()
            if new_mask_only.any():
                self._reset_new_weights(name, new_mask_only.float())
            overlap_mask = desired_mask & prev_bool
            combined_mask = overlap_mask | new_mask_only
            self.mask.masks[name] = combined_mask.to(dtype=mask_tensor.dtype)
        self.mask.fired_masks = copy.deepcopy(self.mask.masks)
        self.mask._apply_mask()

    def _freeze_used_params(self):
        # ENCL isolation: block gradients through connections assigned to earlier tasks.
        if self.cur_task != 0 and self.isolate:
            for (name, params) in self.net.named_parameters():
                if params.grad is None:
                    continue
                used_mask = self.used_params.get(name)
                if used_mask is None:
                    continue
                mask_device = used_mask.to(params.grad.device)
                params.grad[mask_device != 0] = 0

    def _update_used_params(self):
        mask_path = f'{self.masks_dir}/{self.data_manager.dataset}_task{self.cur_task}_mask.pt'
        if not os.path.exists(mask_path):
            print_and_log(f'WARNING: Mask file not found for task {self.cur_task}: {mask_path}')
            return
        task_mask = torch.load(mask_path, map_location=self.device)
        for (name, mask) in task_mask.items():
            if 'normalization' in name:
                continue
            mask_tensor = mask.to(self.device)
            prev = self.used_params.get(name)
            if isinstance(prev, torch.Tensor):
                prev = prev.to(self.device)
                self.used_params[name] = prev + mask_tensor
            else:
                self.used_params[name] = mask_tensor.clone()

    def _stack_subnets(self):
        # ENCL consolidation: merge only the current task's new connections into the dense model.
        was_training = self.net.training
        self.net.eval()
        if self.cur_task == 0 or self.NET is None:
            self.NET = self._get_model(self.data_manager.num_classes)
            self._load_compatible_state_dict(self.NET, self.net.state_dict())
            self.NET = self.NET.to(self.device)
        else:
            dense_path = f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt'
            state = torch.load(dense_path, map_location=self.device)
            self._load_compatible_state_dict(self.NET, state)
            self.NET = self.NET.to(self.device)
            trained_params_snapshot = {name: param.data.detach().clone() for (name, param) in self.net.named_parameters()}
            self._load_compatible_state_dict(self.net, self.NET.state_dict())
            for ((name, param), (old_name, old_param)) in zip(self.net.named_parameters(), self.NET.named_parameters()):
                param.data.copy_(old_param.data.to(param.data.device))
                task_mask_tensor = self.current_task_mask.get(name)
                if task_mask_tensor is not None and isinstance(task_mask_tensor, torch.Tensor) and (task_mask_tensor.shape == param.data.shape):
                    task_mask_bool = task_mask_tensor.to(param.data.device).bool()
                    trained_param = trained_params_snapshot[name].to(param.data.device)
                    if task_mask_bool.any():
                        param.data[task_mask_bool] = trained_param[task_mask_bool]
                    continue
                zero_mask = param.data == 0
                if zero_mask.any():
                    param.data[zero_mask] = old_param.data.to(param.data.device)[zero_mask]
            self.NET = self._get_model(self.data_manager.num_classes)
            self._load_compatible_state_dict(self.NET, self.net.state_dict())
            self.NET = self.NET.to(self.device)
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(self.NET.state_dict(), f'{self.models_dir}/{self.data_manager.dataset}_DENSE.pt')
        if was_training:
            self.net.train()

    def _set_mask(self, task_id):
        # ENCL inference: recover a task-specific subnetwork with its saved mask.
        mask_path = f'{self.masks_dir}/{self.data_manager.dataset}_task{task_id}_mask.pt'
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'Mask not found: {mask_path}')
        task_mask = torch.load(mask_path, map_location=self.device)
        if self.NET is None:
            raise ValueError('Dense model (self.NET) not available. Cannot apply mask. Train a task first.')
        masked_net = self._get_model(self.data_manager.num_classes)
        self._load_compatible_state_dict(masked_net, self.NET.state_dict())
        masked_net = masked_net.to(self.device)
        for (n, t) in masked_net.named_parameters():
            if n in task_mask:
                mask_tensor = task_mask[n].to(self.device) if isinstance(task_mask[n], torch.Tensor) else task_mask[n]
                t.data = t.data * mask_tensor
        return masked_net

    def _save_mask(self):
        os.makedirs(self.masks_dir, exist_ok=True)
        expert_masks_cpu = {}
        for (name, mask_tensor) in self.mask.masks.items():
            if not isinstance(mask_tensor, torch.Tensor):
                continue
            expert_masks_cpu[name] = mask_tensor.detach().clone().cpu()
        with torch.no_grad():
            for (param_name, param) in self.net.named_parameters():
                if param_name.endswith('.bias') and param_name not in expert_masks_cpu:
                    expert_masks_cpu[param_name] = torch.ones_like(param, device='cpu')
        torch.save(expert_masks_cpu, f'{self.masks_dir}/{self.data_manager.dataset}_task{self.cur_task}_mask.pt')
