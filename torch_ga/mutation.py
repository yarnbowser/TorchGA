import torch
from abc import ABC, abstractmethod
from typing import Callable
from functools import partial


class MutationFunction(ABC):
  def __init__(self, mutation_rate: float):
    assert mutation_rate >= 0 and mutation_rate <= 1, 'Mutation rate must be between 0 and 1'
    self.mutation_rate = mutation_rate

  @abstractmethod
  def __call__(self, population: torch.Tensor) -> torch.Tensor:
    pass



class BinaryMutation(MutationFunction):
  def __init__(self, mutation_rate: float):
        super().__init__(mutation_rate)

  def __call__(self, population: torch.Tensor) -> torch.Tensor:
    return population.where(torch.rand_like(population, dtype=torch.float16) >= self.mutation_rate, 1 - population)



# If you define a function to create a randomized population with valid values, this will always work for uniform mutation
class GenericUniformMutation(MutationFunction):
  def __init__(self, mutation_rate: float, population_generator: Callable[..., torch.Tensor], *generator_args, **generator_kwargs):
        super().__init__(mutation_rate)
        self.population_generator = partial(population_generator, *generator_args, **generator_kwargs)
        assert isinstance(self.population_generator(), torch.Tensor), "Population generator must return a torch.Tensor"

  def __call__(self, population: torch.Tensor) -> torch.Tensor:
    mutated_values = self.population_generator()
    assert population.size() == mutated_values.size(), "Population Generator must return a tensor the same size as the population"
    return population.where(torch.rand_like(population, dtype=torch.float16) >= self.mutation_rate, mutated_values)