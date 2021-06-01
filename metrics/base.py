from abc import ABCMeta, abstractmethod
from pathlib import Path


class Metric(metaclass=ABCMeta):

    @abstractmethod
    def update_state(self, model_output, step_data):
        pass

    @abstractmethod
    def log_in_tensorboard(self, step: int = None, reset_state=True):
        pass

    @abstractmethod
    def log_in_stdout(self, step: int = None, reset_state=True):
        pass

    @abstractmethod
    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    def get_values(self, reset_state=True):
        pass


class EmptyMetric(Metric):
    """
    Empty metric that servers as placeholder for metrics if dataset is empty.
    """

    def update_state(self, model_output, step_data):
        raise NotImplementedError("Can't update state of Empty metric!")

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        print("\n\n WARNING: Trying to log EmptyMetric in tensorboard!\n\n")

    def log_in_stdout(self, step: int = None, reset_state=True):
        print("\n\n WARNING: Trying to log EmptyMetric in stdout!\n\n")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        lines = [prepend_str + '\n'] if prepend_str else []
        lines.append(f"WARNING: Empty metrics!\n")

        file_path = Path(file)
        with file_path.open("a") as file:
            file.writelines(lines)

    def reset_state(self):
        raise NotImplementedError("Can't reset state of Empty metric!")
