import torch_ga
from torch_ga import TorchGA
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ones_bit_stream(torch_ga.FitnessFunction):
    def __call__(self, population: torch.Tensor):
        return population.float().mean(dim=-1)
    

pop_size = 200
genome_length = 1000


ga = TorchGA(
    initial_population=(torch.rand(pop_size, genome_length, device=device) > 0.5).int(),
    num_elites=10,
    fitness_function=ones_bit_stream(),
    selection_method='ranked',
    crossover_function='single point',
    mutation_function=torch_ga.mutation.BinaryMutation(mutation_rate=0.001)
)

stats = ga.run_for(4000)

plt.plot(stats.max_fitnesses, label="max")
plt.plot(stats.avg_fitnesses, label="avg")
plt.plot(stats.min_fitnesses, label="min")
plt.title("Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()