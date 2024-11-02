import torch
from abc import ABC, abstractmethod

class PopulationStats(ABC):
    @abstractmethod
    def update(self, population: torch.Tensor, fitnesses: torch.Tensor):
        pass

class FitnessStats(PopulationStats):
    def __init__(self):
        super().__init__()

        self.min_fitnesses = []
        self.max_fitnesses = []
        self.avg_fitnesses = []

    def update(self, population: torch.Tensor, fitnesses: torch.Tensor):
        self.min_fitnesses.append(fitnesses.min(dim=-1))
        self.max_fitnesses.append(fitnesses.max(dim=-1))
        self.avg_fitnesses.append(fitnesses.mean(dim=-1))