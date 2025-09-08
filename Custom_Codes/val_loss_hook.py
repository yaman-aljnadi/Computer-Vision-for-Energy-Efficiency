from mmengine.hooks import Hook
import torch
import os

class ValLossHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        model = runner.model
        model.eval()

        val_loader = runner.val_dataloader
        total_loss = 0.0
        num_samples = 0

        for i, data_batch in enumerate(val_loader):
            with torch.no_grad():
                losses = model.loss(data_batch)
                loss = sum(v.item() for v in losses.values() if isinstance(v, torch.Tensor))
                total_loss += loss
                num_samples += 1

        avg_loss = total_loss / num_samples
        runner.log_buffer.update({'val_loss': avg_loss}, runner.epoch)
        runner.visualizer.add_scalars({'val_loss': avg_loss}, step=runner.epoch, tag='validation')

        model.train()
