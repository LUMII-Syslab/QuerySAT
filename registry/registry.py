from abc import abstractmethod

from data.CNFGen import SAT_3, Clique, KColor
from data.SHAGen2019 import SHAGen2019
from data.k_sat import KSAT
from model.neurocore import NeuroCore
from model.querysat import QuerySAT
from model.neurocore_query import NeuroCoreQuery


class Registry:

    @property
    @abstractmethod
    def registry(self) -> dict:
        pass

    def resolve(self, name):
        if name in self.registry:
            return self.registry.get(name)

        raise ModuleNotFoundError(f"Model with name {name} is not registered!")

    @property
    def registered_names(self):
        return self.registry.keys()


class ModelRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "querysat": QuerySAT,
            "neurocore_query": NeuroCoreQuery,
            "neurocore": NeuroCore
        }


class DatasetRegistry(Registry):

    @property
    def registry(self) -> dict:
        return {
            "ksat": KSAT,
            "kcolor": KColor,
            "3sat": SAT_3,
            "clique": Clique,
            "sha2019": SHAGen2019,
        }
