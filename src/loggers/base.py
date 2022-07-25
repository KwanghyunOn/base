from abc import ABCMeta, abstractmethod


class BaseLogger(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_metrics(self, metrics, **kwargs):
        pass

    @abstractmethod
    def state_dict(self):
        pass


class DummyLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def log_metrics(self, metrics, **kwargs):
        return

    def state_dict(self):
        return {}