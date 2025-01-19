import torch
from einops import repeat


def extract_genomes(population: torch.Tensor, fitnesses: torch.Tensor, fitness_indices: torch.Tensor) -> torch.Tensor:
    O, P, *G = population.size()
    index = fitness_indices.view(O, -1, *[1] * len(G)).repeat(1, 1, *G) # Shape (O, K, G...)
    selected = population.gather(dim=1, index=index)
    return selected, fitnesses.gather(dim=1, index=fitness_indices)



def k_select(population: torch.Tensor, fitnesses: torch.Tensor, k: int) -> torch.Tensor:
    top_fitnesses, top_indices = torch.topk(fitnesses, k=k, dim=1, sorted=False)
    return extract_genomes(population, fitnesses, top_indices)



def sample(population: torch.Tensor, fitnesses: torch.Tensor, probs: torch.Tensor, num_genomes: int, replacement: bool=True) -> torch.Tensor:
    if not replacement and num_genomes > fitnesses.size(1):
        print(f'Warning: Cannot select {num_genomes} genomes from population of size {fitnesses.size(1)} without replacement. Replacement will be used')
        replacement = True

    selection_indices = torch.multinomial(probs, num_genomes, replacement=replacement)
    return extract_genomes(population, fitnesses, selection_indices)