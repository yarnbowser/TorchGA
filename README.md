# TorchGA
A genetic algorithm implementation for tensor-based genomes built on PyTorch.

If you want to evolve multiple objectives in parallel, 
structure your initial population as (n_objectives, pop_size, genome_shape...) 
and pass in `multiple_objectives=True` to the TorchGA

## Selection options:  
Assume we have fitnesses `[f_0, f_1, ..., f_{I-1}]`, 
such that `f_i > f_j` when `i > j`. 
In practice these won't be ordered, but we calculate the probability `p(i)` of genome with fitness `f_i` being selected as one of `n` genomes.  

Since `torch.multinomial` auto-normalizes probabilities, we can simplify our calculation of `p(i)` by removing any constants as long as the simplified probabilities are all positive

### Proportional Selection
General form:
```  
p(i) = f_i / sum(j=[0...I-1] f_j)
```

Simplified form:  
```
if f_i >= 0 for all i
    p(i) = f_i
else
    p(i) = f_i - f_0
```

### Ranked Selection
General form:
```
p(i) = (i + 1) / sum(j=[1...I] j)
```

Simplified form:
```
p(i) = i + 1
```

### Roulette Wheel Selection
General form:
```
p(i) = 1 / I
```

Simplified form:
```
p(i) = 1
```

### Softmax Selection  
params: temperature=T  
```  
=> p(i) = exp(f_i / T) / sum(j=[0...I-1] exp(f_j / T))
```

### Tournament Selection
General form:
```
p(i) = i_C_{k-1} / I_C_k
= i!k!(I-k)! / I!(k-1)!(i-k+1)!
```

Since we're dealing with factorials, we need to scale down the values. Observe that `(I-k)! / I! ~< I^{-k+1}`, so we can divide the probabilities by `I^{-k+2}`. To avoid exploding values from the factorials, we calculate everything in logspace.  

Simplified form:
```
p(i) = i! / (i-k+1)!
= prod(j=[0...k-2] i-j)
= exp(sum(j=[0...k-2] log(i-j)) - log(n) * (k-2))
```

### Truncation Selection
```
p(i) = 1 if i >= I - n else 0
```


