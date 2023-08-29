import logging
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, logs_dir) -> None:
        self.logs_dir = logs_dir

    @abstractmethod
    def watch(self, model):
        """
        Monitor parameters and gradients.
        """
        pass

    def log(self, update_dict, step=None, split: str = ""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        return update_dict

    @abstractmethod
    def log_plots(self, plots):
        pass

    @abstractmethod
    def mark_preempting(self):
        pass


class TensorboardLogger(Logger):
    def __init__(self, logs_dir) -> None:
        super().__init__(logs_dir)
        self.writer = SummaryWriter(logs_dir)

    # TODO: add a model hook for watching gradients.
    def watch(self, model) -> bool:
        logging.warning(
            "Model gradient logging to tensorboard not yet supported."
        )
        return False

    def log(self, update_dict, step=None, split: str = ""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], int) or isinstance(
                    update_dict[key], float
                )
                self.writer.add_scalar(key, update_dict[key], step)

    def mark_preempting(self):
        pass

    def log_plots(self, plots):
        pass
