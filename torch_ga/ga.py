import torch
from typing import Callable, Tuple

from .selection import SelectionMethod
from .mutation import MutationFunction
from .crossover import CrossoverFunction
from .fitness import FitnessFunction
from .stats import PopulationStats, FitnessStats
from .utils import k_select

class TorchGA:
  def __init__(
    self,
    initial_population: torch.Tensor,
    num_elites: int,
    fitness_function: FitnessFunction | Callable[[torch.Tensor], torch.Tensor],
    selection_method: SelectionMethod | Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]] | str,
    crossover_function: CrossoverFunction | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | str,
    mutation_function: MutationFunction | Callable[[torch.Tensor], torch.Tensor],
    population_stats: PopulationStats=FitnessStats(),
    multiple_objectives: bool=False
  ):    
    if isinstance(selection_method, str):
      selection_method = SelectionMethod.get_selection_by_name(selection_method)
    
    if isinstance(crossover_function, str):
      crossover_function = CrossoverFunction.get_crossover_by_name(crossover_function)

    if not multiple_objectives:
      initial_population = initial_population.unsqueeze(0)

    self.population = initial_population
    self.initial_population = initial_population
    self.num_elites = num_elites
    self.fitness_function = fitness_function
    self.crossover_function = crossover_function
    self.mutation_function = mutation_function
    self.selection_method = selection_method
    self.poplation_stats = population_stats

    

    # O, P, *G = Num_Objectives, Pop_Size, Genome_Dims
    O, P, *G = self.population.size()

    assert num_elites > 0 and num_elites < P, "Number of elites must be greater than 0 and less than population size"

    self.fitnesses = fitness_function(self.population)
    self.initial_fitnesses = self.fitnesses

    self.num_offspring = P - num_elites

    self.poplation_stats.update(self.population, self.fitnesses)

  def reset(self):
    self.population = self.initial_population
    self.fitnesses = self.initial_fitnesses
    self.poplation_stats.clear()


  @torch.no_grad()
  def next_generation(self):
    elites, elite_fitnesses = k_select(self.population, self.fitnesses, self.num_elites)
    parents, parent_fitnesses = self.selection_method(self.population, self.fitnesses, num_genomes=self.num_offspring)
    offspring = self.crossover_function(parents, parents.flip(1))
    offspring = self.mutation_function(offspring)
    offspring_fitnesses = self.fitness_function(offspring)

    self.population = torch.cat([elites, offspring], dim=1)
    self.fitnesses = torch.cat([elite_fitnesses, offspring_fitnesses], dim=1)

    return self.population, self.fitnesses

  @torch.no_grad()
  def run_for(self, num_generations):
    for _ in range(num_generations):
      self.next_generation()
      self.poplation_stats.update(self.population, self.fitnesses)

    return self.poplation_stats


