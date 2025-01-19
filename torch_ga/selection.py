import torch
from abc import ABC, abstractmethod
import math
from einops import rearrange

from .utils import sample, k_select


class SelectionMethod(ABC):
  @abstractmethod
  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    pass

  def get_selection_by_name(selection_method: str):
    match selection_method.lower():
      case 'roulette' | 'roulette_wheel':
        return RouletteSelection()
      case 'proportional':
        return ProportionalSelection()
      case 'softmax' | 'boltzmann':
        return SoftmaxSelection()
      case 'ranked':
        return RankedSelection()
      case 'truncation':
        return TruncationSelection()
      case 'tournament':
        return TournamentSelection()
      case _:
        raise NotImplementedError(f'{selection_method} Selection is not implemented')



class RouletteSelection(SelectionMethod):
  def __init__(self, replacement=True):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    return sample(population, fitnesses, torch.ones_like(fitnesses, dtype=torch.double), num_genomes, replacement=self.replacement)



class ProportionalSelection(SelectionMethod):
  def __init__(self, replacement=True):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    probs = fitnesses
    mins, _ = probs.min(dim=1, keepdim=True)
    if torch.any(mins < 0):
      probs -= mins # shift the min to 0 if fitnesses have negatives
    return sample(population, fitnesses, probs.double(), num_genomes, replacement=self.replacement)
  

class SoftmaxSelection(SelectionMethod):
  def __init__(self, temperature: float=1.0, replacement=True):
    self.replacement = replacement
    self.temp = temperature

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    probs = torch.softmax(fitnesses / self.temp, dim=1)
    return sample(population, fitnesses, probs.double(), num_genomes, replacement=self.replacement)



class RankedSelection(SelectionMethod):
  def __init__(self, replacement=True):
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    probs = fitnesses.argsort(dim=1) + 1
    return sample(population, fitnesses, probs.double(), num_genomes, replacement=self.replacement)



class TruncationSelection(SelectionMethod):
  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    return k_select(population, fitnesses, num_genomes)



class TournamentSelection(SelectionMethod):
  def __init__(self, tournament_size: int=4, replacement=True):
    assert tournament_size >= 2, f"Cannot do tournament selection with tournaments of size {tournament_size}, choose a value >= 2"
    fact = torch.arange(tournament_size - 1, dtype=torch.int8)
    self.subtraction = rearrange(fact, 'k -> 1 1 k')
    self.k_minus_2 = tournament_size - 2 if tournament_size > 2 else tournament_size - 1
    self.replacement = replacement

  def __call__(self, population: torch.Tensor, fitnesses: torch.Tensor, num_genomes: int) -> torch.Tensor:
    indices = torch.argsort(fitnesses, dim=-1)

    products = indices.unsqueeze(-1) - self.subtraction.to(population.device)
    products[products < 0] = 0

    log_probs = products.log().sum(dim=-1) # values probably explode if you compute directly
    probs = torch.exp(log_probs - math.log(num_genomes) * self.k_minus_2) # scale in logspace before exponentiating

    return sample(population, fitnesses, probs, num_genomes, replacement=self.replacement)
  