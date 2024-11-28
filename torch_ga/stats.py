import torch
from abc import ABC, abstractmethod


class PopulationStats(ABC):
    @abstractmethod
    def update(self, population: torch.Tensor, fitnesses: torch.Tensor):
        pass

    @abstractmethod
    def clear(self):
        pass



class FitnessStats(PopulationStats):
    def __init__(self):
        super().__init__()

        self.clear()

    def update(self, population: torch.Tensor, fitnesses: torch.Tensor):
        self.min_fitnesses.append(fitnesses.min(dim=-1).values.round(decimals=3).tolist())
        self.max_fitnesses.append(fitnesses.max(dim=-1).values.round(decimals=3).tolist())
        self.avg_fitnesses.append(fitnesses.mean(dim=-1).round(decimals=3).tolist())

    def clear(self):
        self.min_fitnesses = []
        self.max_fitnesses = []
        self.avg_fitnesses = []