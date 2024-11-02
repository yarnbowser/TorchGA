import torch
from abc import ABC, abstractmethod


class FitnessFunction(ABC):
  @abstractmethod
  def __call__(self, population: torch.Tensor) -> torch.Tensor:
    pass