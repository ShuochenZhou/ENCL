import torch

from .base import BaseContinualLearner
from ..common.experiment import (
    clear_gradients,
    plain_sgd_update,
    print_and_log,
)


class LocalLearningLearner(BaseContinualLearner):
    def _current_head_update_parameters(self):
        return list(self._current_mr_task_head().parameters())

    def _clear_current_head_gradients(self, parameters):
        clear_gradients(parameters)

    def _update_current_head(self, parameters):
        plain_sgd_update(parameters, self.args.feature_head_lr)

    def _iter_local_learning_layers(self):
        for module in self.net.modules():
            if module.__class__.__name__ == 'LocalLearningSignalGenerationLayer':
                yield module

    def _reset_local_learning_statistics(self):
        for layer in self._iter_local_learning_layers():
            layer.reset_statistics()

    def _log_local_learning_statistics(self, epoch):
        for idx, layer in enumerate(self._iter_local_learning_layers()):
            stats = layer.get_statistics()
            loss_avg = stats.get('loss_avg')
            samples = stats.get('samples')
            step_count = stats.get('step_count')
            lr = stats.get('lr')
            acc = stats.get('acc')
            pred_hist = stats.get('pred_hist')
            correct_logit_mean = stats.get('correct_logit_mean')
            incorrect_logit_mean = stats.get('incorrect_top_logit_mean')
            loss_str = f'{loss_avg:.6f}' if isinstance(loss_avg, (int, float)) else 'n/a'
            samples_str = str(samples) if samples is not None else 'n/a'
            steps_str = str(step_count) if step_count is not None else 'n/a'
            lr_str = f'{lr:.6f}' if isinstance(lr, (int, float)) else 'n/a'
            acc_str = f'{acc * 100:.2f}%' if isinstance(acc, (int, float)) else 'n/a'
            correct_logit_str = f'{correct_logit_mean:.4f}' if isinstance(correct_logit_mean, (int, float)) else 'n/a'
            incorrect_logit_str = f'{incorrect_logit_mean:.4f}' if isinstance(incorrect_logit_mean, (int, float)) else 'n/a'
            top_str = 'n/a'
            if pred_hist is not None:
                counts = (
                    pred_hist.tolist()
                    if isinstance(pred_hist, torch.Tensor)
                    else list(pred_hist)
                )
                if sum(counts) > 0:
                    top_entries = sorted(
                        (
                            (class_index, count)
                            for class_index, count in enumerate(counts)
                            if count > 0
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                    top_str = ', '.join(
                        f'{class_index}:{count}'
                        for class_index, count in top_entries
                    )
            print_and_log(
                f'  Block {idx}: loss_avg={loss_str}, acc={acc_str}, '
                f'samples={samples_str}, update_steps={steps_str}, '
                f'lr={lr_str}, top_preds={top_str}, '
                f'logit_target={correct_logit_str}, '
                f'logit_top_other={incorrect_logit_str}'
            )
