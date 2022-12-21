import torch
from src.engine.core import Hooks
from src.config.registry import Registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINER_REGISTRY = Registry("TRAINER")


class ModelTemplate(Hooks):
    def __init__(self):
        self.model = None

    def before_train(self):
        self.model


def build_trainer(cfg, collector, loger={}):
    name = cfg.TRAIN.TRAINER
    trainer_cls = TRAINER_REGISTRY.get(name)
    return trainer_cls(cfg.TRAIN, collector, {})


