import torch
from typing import Callable

from selection import SelectionMethod
from mutation import MutationFunction
from crossover import CrossoverFunction
from fitness import FitnessFunction
from stats import PopulationStats, FitnessStats

class VectorizedGA:
  def __init__(
    self,
    initial_population: torch.Tensor,
    mutation_rate: float,
    num_elites: int,
    selection_method: SelectionMethod | Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    fitness_function: FitnessFunction | Callable[[torch.Tensor], torch.Tensor],
    crossover_function: CrossoverFunction | Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    mutation_function: MutationFunction | Callable[[torch.Tensor], torch.Tensor],
    population_stats: PopulationStats=FitnessStats()
  ):

    assert initial_population.dim() == 2 or initial_population.dim() == 3, 'Population must be in the shape of (Pop_Size, Genome_Length) or (Num_Objectives, Pop_Size, Genome_Length)'
    assert mutation_rate >= 0 and mutation_rate <= 1, 'Mutation rate must be between 0 and 1'
    assert type(fitness_function = FitnessFunction), "Fitness function must extend abstract type FitnessFunction"
    assert type(crossover_function = CrossoverFunction), "Crossover function must extend abstract type CrossoverFunction"
    assert type(mutation_function = MutationFunction), "Mutation function must extend abstract type MutationFunction"
    assert type(population_stats = PopulationStats), "Stats object function must extend abstract type PopulationStats"


    self.population = initial_population
    self.mutation_rate = mutation_rate
    self.num_elites = num_elites
    self.fitness_function = fitness_function
    self.crossover_function = crossover_function
    self.mutation_function = mutation_function
    self.selection_method = selection_method
    self.poplation_stats = population_stats

    if self.population.dim() == 2:
      self.population = self.population.unsqueeze(0) # Add the objective dimension if none was given

    # O, P, G = Num_Objectives, Pop_Size, Genome_length
    O, P, G = self.population.size()

    assert num_elites > 0 and num_elites < P, "Number of elites must be greater than 0 and less than population size"

    self.fitnesses = fitness_function(self.population)
    self.num_offspring = P - num_elites

    self.poplation_stats.update(self.population, self.fitnesses)


  def next_generation(self):
    elites, elite_fitnesses = SelectionMethod.k_select(self.population, self.fitnesses, self.num_elites)
    parents = self.selection_method(self.population, self.fitnesses, num_genomes=self.num_offspring)

    offspring = self.crossover_function(parents)
    offspring = self.mutation_function(offspring)
    offspring_fitnesses = self.fitness_function(offspring)

    self.population = torch.cat([elites, offspring], dim=-2)
    self.fitnesses = torch.cat([elite_fitnesses, offspring_fitnesses], dim=-1)

    return self.population, self.fitnesses


  def run_for(self, num_generations):
    for _ in range(num_generations):
      self.next_generation()
      self.poplation_stats.update(self.population, self.fitnesses)

    return self.poplation_stats

    
