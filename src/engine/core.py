

class Base:

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    def before_batch(self, *args, **kwargs):
        pass

    def after_batch(self, *args, **kwargs):
        pass


class Hooks:
    def __init__(self, cfg):
        self.cfg = cfg

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    def before_batch(self, *args, **kwargs):
        pass

    def after_batch(self, *args, **kwargs):
        pass
