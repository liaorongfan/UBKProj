import os
import torch
from src.engine.core import Hooks


class Checkpointer(Hooks):
    def __init__(self, cfg):
        super().__init__(cfg)

    def before_train(self, model, optimizer, epoch):
        if self.cfg.TRAIN.RESUME:
            checkpoint_path = self.cfg.TRAIN.RESUME
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
        print("Train from scratch ... ")

    def after_epoch(self, model, optimizer, epoch, best_acc, *args, **kwargs):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }
        pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch != (self.cfg.MAX_EPOCH - 1) else "checkpoint_last.pkl"
        output_dir = self.cfg.TRAIN.OUTPTU_DIR
        path_checkpoint = os.path.join(output_dir, pkl_name)
        torch.save(checkpoint, path_checkpoint)

