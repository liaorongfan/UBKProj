from .core import Base
from .construct import Constructor
from src.checkpoint.checkpointer import Checkpointer
from src.engine.build import device


class Template(Constructor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._hooks = []

    def register_hooks(self, hook_obj):
        self._hooks.append(hook_obj)

    def before_train(self, *args, **kwargs):
        for hook in self._hooks:
            hook.before_train(*args, **kwargs)

    def after_train(self, *args, **kwargs):
        for hook in self._hooks:
            hook.after_train(*args, **kwargs)

    def before_epoch(self, *args, **kwargs):
        for hook in self._hooks:
            hook.before_epoch(*args, **kwargs)

        # lr = self.optimizer.param_groups[0]['lr']
        # self.logger.info(f"Training: learning rate:{lr}")

    def after_epoch(self, epoch, *args, **kwargs):
        for hook in self._hooks:
            hook.after_epoch(*args, **kwargs)

        # self.scheduler.step()
        #
        # if self.collector.model_save and epoch % self.cfg.VALID_INTERVAL == 0:
        #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, self.cfg)
        #     self.collector.update_best_epoch(epoch)
        # if self.cfg.TRAIN.SAVE_LAST and epoch == (self.cfg.MAX_EPOCH - 1):
        #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, self.cfg)

    def before_batch(self,  *args, **kwargs):
        for hook in self._hooks:
            hook.before_batch(*args, **kwargs)

    def after_batch(self,  *args, **kwargs):
        for hook in self._hooks:
            hook.after_batch(*args, **kwargs)

    def run_epoch(self):
        for i, data in enumerate(self.data_loader["train"]):
            self.before_batch()
            self.run_batch(data)
            self.after_batch()

    def run_batch(self, data):
        inputs, labels = data["data"].to(device), data["label"].to(device)
        outputs = self.model(inputs)
        loss = self.loss_f(outputs.cpu(), labels.cpu())
        print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        self.before_train(self.model, self.optimizer, self.epoch)
        for epoch in range(self.cfg.TRAIN.START_EPOCH, self.cfg.TRAIN.MAX_EPOCH):
            self.before_epoch(epoch)
            self.run_epoch()
            self.after_epoch()
        self.after_train()


class TemplateTrainer(Template):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.build()
        self.register_hooks()

    def register_hooks(self):
        checkpointer = Checkpointer(self.cfg)
        self._hooks = [
            checkpointer,
            # self.trainer, self.collector,
            # self.optimizer, self.scheduler,
        ]

    def test(self):
        pass



