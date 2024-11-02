import torch
from abc import ABC, abstractmethod
import math
from einops import rearrange

from utils import sample, k_select

class SelectionMethod(ABC):
  @abstractmethod
  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    pass



class RouletteSelection(SelectionMethod):
  def __init__(self, replacement=False):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    return sample(population, fitnesses, torch.ones_like(fitnesses, dtype=torch.float), num_genomes, replacement=self.replacement)



class ProportionalSelection(SelectionMethod):
  def __init__(self, replacement=False):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    probs = fitnesses
    mins, _ = probs.min(dim=-1, keepdim=True)
    if (mins < 0).any():
      probs -= mins # shift the min to 0 if fitnesses have negatives
    return sample(population, fitnesses, probs, num_genomes, replacement=self.replacement)

class RankedSelection(SelectionMethod):
  def __init__(self, replacement=False):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    probs = torch.argsort(fitnesses, dim=-1) + 1
    return sample(population, fitnesses, probs, num_genomes, replacement=self.replacement)



class TruncationSelection(SelectionMethod):
  def __init__(self, replacement=False):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    return k_select(population, fitnesses, num_genomes)


class TournamentSelection(SelectionMethod):
  def __init__(self, tournament_size: int, replacement=True):
    self.subtraction = rearrange(torch.arange(tournament_size - 1), 'k -> 1 1 k')
    self.k_minus_2 = tournament_size - 2
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    indices = torch.argsort(fitnesses, dim=-1)

    products = indices.unsqueeze(-1) - self.subtraction.to(population.device)
    products[products < 0] = 0

    log_probs = products.log().sum(dim=-1) # values probably explode if you compute directly
    probs = torch.exp(log_probs - math.log(num_genomes) * self.k_minus_2) # scale in logspace before exponentiating

    return sample(population, fitnesses, probs, num_genomes, replacement=self.replacement)
  