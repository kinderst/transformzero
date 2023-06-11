"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from utils.min_gpt_utils import CfgNode


class Trainer:

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, val_dataset, save_weights_path=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_weights_path = save_weights_path  # string where to save weights i.e. "best_model_path.pth"
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self, log_level=0):
        model, config = self.model, self.config

        # set up the optimizer
        self.optimizer = model.configure_optimizers(config)

        # set up the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # set up the validation dataloader
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Validation
            if self.iter_num % config.val_interval == 0:
                if log_level:
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    self.iter_time = tnow
                    print(f"iter_dt {self.iter_dt * 1000:.2f}ms; iter {self.iter_num}: train loss {self.loss.item():.5f}")

                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    val_losses = []
                    for val_batch in val_loader:
                        val_batch = [t.to(self.device) for t in val_batch]
                        val_x, val_y = val_batch
                        val_logits, val_loss = model(val_x, val_y)
                        val_losses.append(val_loss.item())
                    average_val_loss = sum(val_losses) / len(val_losses)
                    if log_level:
                        print("avg val loss: ", average_val_loss)

                # Update the best validation loss and check for improvement
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    epochs_without_improvement = 0
                    # Save the model weights
                    if self.save_weights_path:
                        torch.save(model.state_dict(), self.save_weights_path)
                else:
                    epochs_without_improvement += 1

                # Terminate if validation loss doesn't improve for a certain number of epochs
                if epochs_without_improvement >= config.patience:
                    break

            # iter termination conditions
            self.iter_num += 1

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
