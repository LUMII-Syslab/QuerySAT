from abc import abstractmethod, ABCMeta

import tensorflow as tf


class Dataset(metaclass=ABCMeta):
    """ Base dataset that other datasets must implement to be compliant
    with training framework.
    """

    @abstractmethod
    def train_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def validation_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def test_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def filter_model_inputs(self, step_data) -> dict:
        pass

    @abstractmethod
    def metrics(self, initial=False) -> list:
        pass
