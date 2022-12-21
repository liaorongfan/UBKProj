from .core import Base
from src.data.build import build_dataloader
from src.modeling.network.build import build_model
from src.modeling.loss.build import build_loss_func
from src.modeling.solver.build import build_solver, build_scheduler
from src.engine.build import build_trainer
from src.evaluate.summary import TrainSummary


class Constructor(Base):
    """
    construct certain experiment by the following template
    step 1: prepare dataloader
    step 2: prepare model and loss function
    step 3: select optimizer for gradient descent algorithm
    step 4: prepare trainer for typical training in pytorch manner
    """
    def __init__(self, cfg):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """
        self.cfg = cfg
        self.data_loader = None
        self.model = None
        self.loss_f = None
        self.optimizer = None
        self.scheduler = None
        self.collector = None
        self.trainer = None
        self.epoch = 0

    def build(self):
        self.data_loader = self.build_dataloader()
        self.model = self.build_model()
        self.loss_f = self.build_loss_function()
        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()
        self.collector = TrainSummary()
        # self.trainer = self.build_trainer()

    def build_dataloader(self):
        return build_dataloader(self.cfg)

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    # def build_trainer(self):
    #     return build_trainer(self.cfg, self.collector)
