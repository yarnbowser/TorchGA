import torch
from abc import ABC, abstractmethod


class CrossoverFunction(ABC):
  def __init__(self, crossover_rate):
    self.crossover_rate = crossover_rate

  @abstractmethod
  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    pass


class InterleavingCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=0.5):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    return first_parents.where(torch.rand_like(first_parents, dtype=torch.float16) <= self.crossover_rate, second_parents)


class SinglePointCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=1):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    *B, P, _ = first_parents.size()
    skip_crossover = torch.rand(*B, P, 1, device=first_parents.device) >= self.crossover_rate
    point = torch.randint(0, P, (*B, P, 1), device=first_parents.device)
    crossover_index = point >= torch.arange(0, P, device=first_parents.device).unsqueeze(-1)
    return first_parents.where(skip_crossover | crossover_index, second_parents)


class DoublePointCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=1):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    *B, P, _ = first_parents.size()
    skip_crossover = torch.rand(*B, P, 1, device=first_parents.device) >= self.crossover_rate
    first_point = torch.randint(0, P, (*B, P, 1), device=first_parents.device)
    second_point = torch.randint(0, P, (*B, P, 1), device=first_parents.device)
    indices = torch.arange(0, P, device=first_parents.device).unsqueeze(-1)
    crossover_index = (first_point <= indices) & (second_point >= indices)
    return first_parents.where(skip_crossover | crossover_index, second_parents)
