import torch_ga
from torch_ga import TorchGA
import torch
import matplotlib.pyplot as plt
import math
from einops import repeat, reduce
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fast_fitness_function(torch_ga.FitnessFunction):
    def __call__(self, population: torch.Tensor):
        return population.float().mean(dim=-1)
    

class slow_fitness_function(torch_ga.FitnessFunction):
    def __call__(self, population: torch.Tensor):
        fitnesses = []
        for i in range(population.size(-2)):
            fitnesses.append(reduce(population[..., i, :].float(), '... G -> ...', reduction='mean'))
        return torch.stack(fitnesses, dim=-1)
    

pop_size = 200
genome_length = 1000


initial_pop = (torch.rand(pop_size, genome_length, device=device) > 0.5).int()

ga = TorchGA(
    initial_population=initial_pop,
    num_elites=15,
    fitness_function=fast_fitness_function(),
    selection_method='ranked',
    crossover_function='interleaving',
    mutation_function=torch_ga.mutation.BinaryMutation(mutation_rate=0.002)
)



@torch.no_grad()
def run_random(ga, num_generations, num_interleaved):
    num_normal_steps = num_generations // num_interleaved - 2
    random_selection = torch_ga.selection.RouletteSelection()

    for gen in range(num_generations):
        if gen % num_normal_steps or num_interleaved == 0:
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


@torch.no_grad()
def run_sorted(ga, num_generations, num_interleaved):
    num_normal_steps = num_generations // num_interleaved - 2

    for gen in range(num_generations):
        if gen % num_normal_steps or num_interleaved == 0:
            elites = ga.population[..., :ga.num_elites, :]
            elite_fits = ga.fitnesses[..., :ga.num_elites]

            probs = torch.arange(ga.population.size(-2), 0, -1).to(device, torch.double)
            selection_indices = torch.multinomial(probs, ga.num_offspring, replacement=True)

            parents, parent_fitnesses = torch_ga.utils.extract_genomes(ga.population, ga.fitnesses, selection_indices)
            offspring = ga.crossover_function(parents, parents.flip(-2))
            offspring = ga.mutation_function(offspring)
            offspring_fitnesses = ga.fitness_function(offspring) # in practice you wouldn't look at fitnesses, this is just for testing

            offspring_rank = selection_indices + selection_indices.flip(-1)
            offspring_indices = offspring_rank.argsort(dim=-1)

            offspring, offspring_fitnesses = torch_ga.utils.extract_genomes(offspring, offspring_fitnesses, offspring_indices)

            # We maintain our elites to avoid doing worse than when we started. With only a few elites, the best offspring have high probs
            ga.population = torch.cat([elites, offspring], dim=-2)
            ga.fitnesses = torch.cat([elite_fits, offspring_fitnesses], dim=-1)

        else:
            ga.next_generation()

        ga.poplation_stats.update(ga.population, ga.fitnesses)
    return ga.poplation_stats



@torch.no_grad()
def run_fitness_samples(ga, num_generations, full_fitnesses=True):
    # probably should make sorting a util
    sorted_indices = ga.fitnesses.argsort(dim=-1)
    ga.population, ga.fitnesses = torch_ga.utils.extract_genomes(ga.population, ga.fitnesses, sorted_indices)

    for gen in range(num_generations):
        # Keep elites in the front to preserve estimated sort
        elites = ga.population[..., :ga.num_elites, :]
        elite_fits = ga.fitnesses[..., :ga.num_elites]

        # Since population is probably sorted, we just use a reversed range for the ranked selection
        # Not actually a rank, more of a score since first rank has the highest probability
        probs = torch.arange(ga.population.size(-2), 0, -1).to(device, torch.double)
        selection_indices = torch.multinomial(probs, ga.num_offspring, replacement=True)

        # normal ga stuff but we can't use the fitnesses
        parents = ga.population.gather(dim=-2, index=repeat(selection_indices, '... P -> ... P G', G=ga.population.size(-1)))
        offspring = ga.crossover_function(parents, parents.flip(-2))
        offspring = ga.mutation_function(offspring)

        # instead of calculating fitness, we estimate that the offspring score is the average of the parents
        offspring_rank = ga.population.size(-2) - 0.5 * (selection_indices + selection_indices.flip(-1))

        # use the predicted scores to sample a fraction of the offspring for fitness calculation
        fitness_indices = torch.multinomial(offspring_rank, ga.num_elites, replacement=False)
        selected_offspring, _ = torch_ga.utils.extract_genomes(offspring, offspring_rank, fitness_indices)
        selected_fits = ga.fitness_function(selected_offspring)
        
        # separate the offspring that we didn't calculate fitnesses for
        all_indices = torch.arange(offspring.size(-2), device=offspring.device)
        complement_indices = all_indices[~torch.isin(all_indices, fitness_indices)]
        other_offspring, other_ranks = torch_ga.utils.extract_genomes(offspring, offspring_rank, complement_indices)
        
        # sort the excluded offspring based on their predicted score
        offspring_indices = other_ranks.argsort(dim=-1)
        offspring, _ = torch_ga.utils.extract_genomes(other_offspring, other_ranks, offspring_indices)

        

        # sort the fitness calculated offspring with the elites to preserve improvements
        new_elites = torch.cat([elites, selected_offspring], dim=-2)
        new_fits = torch.cat([elite_fits, selected_fits], dim=-1)
        elite_indices = new_fits.argsort(dim=-1, descending=True)
        elites, elite_fits = torch_ga.utils.extract_genomes(new_elites, new_fits, elite_indices)

        # recombine the population (and fitnesses for testing)
        ga.population = torch.cat([elites, offspring], dim=-2)

        if full_fitnesses:
            offspring_fitnesses = ga.fitness_function(offspring) # in practice you wouldn't calculate these fitnesses, this is just for testing
            ga.fitnesses = torch.cat([elite_fits, offspring_fitnesses], dim=-1)
        else:
            ga.fitnesses = elite_fits

        ga.poplation_stats.update(ga.population, ga.fitnesses)
    return ga.poplation_stats


gens = 10000


# stats = run_sorted(ga, gens, math.log(gens))
# sorted_fits = stats.max_fitnesses
# ga.reset()
# stats = run_random(ga, gens, math.log(gens))
# random_fits = stats.max_fitnesses
# ga.reset()
stats = run_fitness_samples(ga, gens)
sampled_fits = stats.max_fitnesses
ga.reset()
stats = ga.run_for(gens)
normal_fits = stats.max_fitnesses
ga.reset()

ga.fitness_function = slow_fitness_function()
normal_times = []
sampled_times = []
generations = range(1000, 6000, 1000)
for i in generations:
    print(i)
    # i'm not running on gpu, synchronize is required if you do though
    start = time.perf_counter()
    ga.run_for(i)
    normal_times.append(time.perf_counter() - start)
    ga.reset()

    start = time.perf_counter()
    run_fitness_samples(ga, i, full_fitnesses=False)
    sampled_times.append(time.perf_counter() - start)
    ga.reset()



# plt.plot(random_fits, label="Total Random Selection", alpha=0.65)
# plt.plot(sorted_fits, label="Pseudo Ranked Selection", alpha=0.65)


fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

axes[0].plot(sampled_fits, label="Partial Fitness Sampling")
axes[0].plot(normal_fits, label="Ranked Selection")
axes[0].set_title("Fitness per Generation")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Fitness")
axes[0].legend()

axes[1].plot(generations, sampled_times, label="Partial Fitness Sampling")
axes[1].plot(generations, normal_times, label="Ranked Selection")
axes[1].set_title("Time for n Generations")
axes[1].set_xlabel("Generations")
axes[1].set_ylabel("Time (s)")
axes[1].legend()

plt.tight_layout()
plt.show()
