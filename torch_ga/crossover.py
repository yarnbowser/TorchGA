import torch
from abc import ABC, abstractmethod


class CrossoverFunction(ABC):
  def __init__(self, crossover_rate):
    self.crossover_rate = crossover_rate

  @abstractmethod
  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    pass

  def get_crossover_by_name(crossover_function: str):
    match crossover_function.lower():
      case 'interleaving' | 'uniform':
        return InterleavingCrossover()
      case 'single_point' | 'single point':
        return SinglePointCrossover()
      case 'double_point' | 'double point':
        return DoublePointCrossover()
      case _:
        raise NotImplementedError(f'{crossover_function} Crossover is not implemented')



class InterleavingCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=0.5):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    return first_parents.where(torch.rand_like(first_parents, dtype=torch.float16) >= self.crossover_rate, second_parents)



class SinglePointCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=1.0):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    O, P, *G = first_parents.size()
    device = first_parents.device

    n_dims = len(G)
    expanded_shape = [1] * n_dims
    g = torch.tensor(G, device=device)

    # Skip crossover with probability (1 - crossover_rate)
    skip_crossover = torch.rand(O, P, *expanded_shape, device=device) > self.crossover_rate

    # We only cat along a single dim in the nD case
    dims = torch.randint(n_dims, (O, P), device=device)
    dim_lens = g[dims]

    # Correct behavior
    # point = torch.cat([torch.randint(1, dim_len, (1,), device=device) for dim_len in dim_lens.flatten().tolist()]).view(O, P)

    # Vectorized Approach
    # We can't give randint a tensor, so we give it the max len and use %
    # We % by 1 - the dim_len then add 1 to get [1, dim_len] instead of [0, dim_len]
    # Not exactly uniform but as close as possible
    points = torch.randint(dim_lens.max(), (O, P), device=device) % (dim_lens - 1) + 1

    # Get aranges for the dims in G...
    ranges = torch.arange(max(G), device=device).expand(n_dims, -1) # Shape: (n_dims, max_dim)
    ranges = ranges[ranges < g.unsqueeze(-1)] # Shape: (sum(G))

    # Cartesian product of the aranges
    m = torch.meshgrid(ranges.split(G), indexing='ij')
    m = torch.stack(m, dim=0) # Shape: (O, P, G...)

    # Create the mask along the chosen dims
    mask = m[dims] < points.view(O, P, *expanded_shape)

    return first_parents.where(skip_crossover | mask, second_parents)


class DoublePointCrossover(CrossoverFunction):
  def __init__(self, crossover_rate=1):
    super().__init__(crossover_rate)

  def __call__(self, first_parents: torch.Tensor, second_parents: torch.Tensor) -> torch.Tensor:
    O, P, *G = first_parents.size()
    device = first_parents.device

    n_dims = len(G)
    expanded_shape = [1] * n_dims
    g = torch.tensor(G, device=device)

    # Skip crossover with probability (1 - crossover_rate)
    skip_crossover = torch.rand(O, P, *expanded_shape, device=device) > self.crossover_rate

    # We only cat along a single dim in the nD case
    dims = torch.randint(n_dims, (O, P), device=device)
    dim_lens = g[dims]

    # We use range [1, dim_len - 1] ensure double point
    points = torch.randint(dim_lens.max() - 1, (2, O, P), device=device) % (dim_lens - 2) + 1
    first_points = points.min(dim=0).values
    second_points = points.max(dim=0).values
    
    # Get aranges for the dims in G...
    ranges = torch.arange(max(G), device=device).expand(n_dims, -1) # Shape: (n_dims, max_dim)
    ranges = ranges[ranges < g.unsqueeze(-1)] # Shape: (sum(G))

    # Cartesian product of the aranges
    m = torch.meshgrid(ranges.split(G), indexing='ij')
    m = torch.stack(m, dim=0)[dims] # Shape: (O, P, G...)

    # Create the mask along the chosen dims
    mask = (m < first_points.view(O, P, *expanded_shape)) | (m > second_points.view(O, P, *expanded_shape))

    return first_parents.where(skip_crossover | mask, second_parents)
