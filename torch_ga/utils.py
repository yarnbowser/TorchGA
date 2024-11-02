import torch
from einops import repeat


def extract_genomes(population: torch.Tensor, fitnesses: torch.Tensor, fitness_indices: torch.Tensor) -> torch.Tensor:
    *_, G = population.size()
    selected = population.gather(dim=-2, index=repeat(fitness_indices, '... P -> ... P G', G=G))
    return selected, fitnesses.gather(dim=-1, index=fitness_indices)



def k_select(population: torch.Tensor, fitnesses: torch.Tensor, k: int) -> torch.Tensor:
    top_fitnesses, top_indices = torch.topk(fitnesses, k=k, sorted=False)
    return extract_genomes(population, fitnesses, top_indices)



def sample(population: torch.Tensor, fitnesses: torch.Tensor, probs: torch.Tensor, num_genomes: int, replacement: bool=True) -> torch.Tensor:
    selection_indices = torch.multinomial(probs, num_genomes, replacement=replacement)
    return extract_genomes(population, fitnesses, selection_indices)