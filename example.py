from einops import reduce
import torch_ga
from torch_ga import TorchGA
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ones_bit_tensor(torch_ga.FitnessFunction):
    def __call__(self, population: torch.Tensor):
        return reduce(population.float(), 'O P ... -> O P', reduction='mean')
    

pop_size = 200
genome_length = [12, 15, 10]


ga = TorchGA(
    initial_population=(torch.rand(pop_size, *genome_length, device=device) > 0.5).int(),
    num_elites=10,
    fitness_function=ones_bit_tensor(),
    selection_method='tournament',
    crossover_function='double point',
    mutation_function=torch_ga.mutation.BinaryMutation(mutation_rate=0.001)
)

stats = ga.run_for(1000)

plt.plot(stats.max_fitnesses, label="max")
plt.plot(stats.avg_fitnesses, label="avg")
plt.plot(stats.min_fitnesses, label="min")
plt.title("Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()