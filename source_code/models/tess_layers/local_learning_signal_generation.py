import torch
from torch import nn
from torch.nn import functional as F

from ...common.experiment import clear_gradients, plain_sgd_update

class LocalLearningSignalGenerationLayer(nn.Module):
    def __init__(self, block, lr, n_classes, hidden_dim, pooling_size):
        super().__init__()
        self.block = block
        self.lr = lr
        self.n_classes = n_classes
        self.temperature = 0.1
        self.label_smoothing = 0.0
        self.pooling_size = pooling_size
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=True)
        nn.init.kaiming_uniform_(self.classifier.weight, a=5 ** 0.5)
        nn.init.zeros_(self.classifier.bias)
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0
        self.step_count = 0
        self.correct = 0
        self.acc_avg = 0
        self.pred_hist = torch.zeros(n_classes, dtype=torch.long)
        self.correct_logit_sum = 0.0
        self.correct_logit_count = 0
        self.incorrect_top_logit_sum = 0.0
        self.incorrect_top_logit_count = 0

    def record_statistics(self, loss, batch_size, logits, targets):
        self.loss_hist += loss.item() * batch_size
        self.samples += batch_size
        self.loss_avg = self.loss_hist / self.samples
        logits_det = logits.detach()
        preds = logits_det.max(dim=1).indices
        self.pred_hist += torch.bincount(
            preds.detach().cpu(), minlength=self.n_classes
        )
        targets_idx = targets.detach().to(logits_det.device).long()
        self.correct += (preds == targets_idx).sum().item()
        batch_indices = torch.arange(logits_det.size(0), device=logits_det.device)
        correct_logits = logits_det[batch_indices, targets_idx]
        self.correct_logit_sum += correct_logits.sum().item()
        self.correct_logit_count += correct_logits.numel()
        masked = logits_det.clone()
        masked[batch_indices, targets_idx] = float("-inf")
        top_incorrect = masked.max(dim=1).values
        valid = torch.isfinite(top_incorrect)
        if valid.any():
            self.incorrect_top_logit_sum += top_incorrect[valid].sum().item()
            self.incorrect_top_logit_count += valid.sum().item()
        self.acc_avg = self.correct / self.samples

    def reset_statistics(self):
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0
        self.step_count = 0
        self.correct = 0
        self.acc_avg = 0
        self.pred_hist.zero_()
        self.correct_logit_sum = 0.0
        self.correct_logit_count = 0
        self.incorrect_top_logit_sum = 0.0
        self.incorrect_top_logit_count = 0

    def clear_gradients(self):
        clear_gradients(self.parameters())

    def apply_plain_sgd(self):
        plain_sgd_update(self.parameters(), self.lr)
        self.step_count += 1

    def get_statistics(self):
        correct_mean = (
            self.correct_logit_sum / self.correct_logit_count
            if self.correct_logit_count > 0 else None
        )
        incorrect_mean = (
            self.incorrect_top_logit_sum / self.incorrect_top_logit_count
            if self.incorrect_top_logit_count > 0 else None
        )
        return {
            "loss_avg": self.loss_avg,
            "samples": self.samples,
            "step_count": self.step_count,
            "lr": float(self.lr),
            "acc": self.acc_avg,
            "pred_hist": self.pred_hist.clone(),
            "correct_logit_mean": correct_mean,
            "incorrect_top_logit_mean": incorrect_mean,
        }

    def _flatten_activation(self, activation):
        batch_size = activation.size(0)
        return F.adaptive_avg_pool2d(
            activation, (self.pooling_size, self.pooling_size)
        ).view(batch_size, -1)

    def forward(self, x, labels=None):
        if not self.training or labels is None:
            return self.block(x)
        out = self.block(x.detach())
        logits = self.classifier(self._flatten_activation(out)) / self.temperature
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss.backward()
        self.record_statistics(loss.detach(), x.size(0), logits, labels)
        return out.detach()
