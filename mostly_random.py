import torch_ga
from torch_ga import TorchGA
import torch
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ones_bit_stream(torch_ga.FitnessFunction):
    def __call__(self, population: torch.Tensor):
        return population.float().mean(dim=-1)
    

pop_size = 200
genome_length = 1000


initial_pop = (torch.rand(pop_size, genome_length, device=device) > 0.5).int()

ga = TorchGA(
    initial_population=initial_pop,
    num_elites=10,
    fitness_function=ones_bit_stream(),
    selection_method='ranked',
    crossover_function='interleaving',
    mutation_function=torch_ga.mutation.BinaryMutation(mutation_rate=0.001)
)



@torch.no_grad()
def run_random(ga, num_generations):
    num_normal_steps = num_generations // (math.log(num_generations)) - 2
    random_selection = torch_ga.selection.RouletteSelection()

    for gen in range(num_generations):
        if gen % num_normal_steps:
            elites = ga.population[..., :ga.num_elites, :]
            elite_fits = ga.fitnesses[..., :ga.num_elites]
            parents, parent_fitnesses = random_selection(ga.population, ga.fitnesses, num_genomes=ga.num_offspring)
            offspring = ga.crossover_function(parents, parents.flip(-2))
            offspring = ga.mutation_function(offspring)
            offspring_fitnesses = ga.fitness_function(offspring) # in practice you wouldn't look at fitnesses, this is just for testing

            ga.population = torch.cat([elites, offspring], dim=-2)
            ga.fitnesses = torch.cat([elite_fits, offspring_fitnesses], dim=-1)
        else:
            ga.next_generation()

        ga.poplation_stats.update(ga.population, ga.fitnesses)
    return ga.poplation_stats




stats = run_random(ga, 5000)
random_fits = stats.max_fitnesses
ga.reset()
stats = ga.run_for(5000)

plt.plot(random_fits, label="Mostly Random Selection")
plt.plot(stats.max_fitnesses, label="Ranked Selection")
plt.title("Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()