# GrowingPartition Ablation: Eager vs Lazy Growth

## Setup
- Problems: 2D multimodal, 10D Gaussian mixture
- Seeds: 42, 123, 456
- Iterations: 100K
- bin_width values: 0.1, 0.2, 0.5, 1.0
- Methods: eager, lazy (threshold=5), lazy (threshold=20), baseline (UniformPartition with known range)

## Key Results

### 2D Multimodal
| Method   | BW  | Best Energy | Flatness | Acc Rate | Bins |
|----------|-----|-------------|----------|----------|------|
| baseline | --  | -8.120      | 0.506    | 0.211    | 42   |
| eager    | 0.1 | -8.120      | 0.799    | 0.204    | 85   |
| eager    | 0.2 | -8.120      | 0.737    | 0.213    | 45   |
| eager    | 0.5 | -8.124      | 0.525    | 0.241    | 21   |
| lazy_20  | 0.2 | -8.121      | 0.754    | 0.218    | 44   |

All strategies find the global minimum (-8.12). GrowingPartition with bw=0.1-0.2 achieves **better flatness** than the hand-tuned baseline (0.74-0.80 vs 0.51).

### 10D Gaussian Mixture
| Method   | BW  | Best Energy | Flatness | Acc Rate | Bins |
|----------|-----|-------------|----------|----------|------|
| baseline | --  | 48.614      | 0.000    | 0.000    | 42   |
| eager    | 0.5 | 0.094       | 0.535    | 0.804    | 140  |
| lazy_20  | 0.5 | 0.083       | 0.583    | 0.797    | 129  |
| eager    | 1.0 | 0.202       | 0.489    | 0.784    | 89   |

The baseline with wrong energy range (e_min=-15, e_max=5) fails completely (best_energy=48.6, 0% acceptance). GrowingPartition automatically discovers the correct range and achieves good optimization.

**bin_width=0.1 is a trap**: it hits max_bins=200 and has terrible flatness (-13). Too many bins for the iteration budget.

**bin_width=0.5** is the sweet spot for 10D: good energy, good flatness, reasonable bin count.

## Eager vs Lazy

There is no meaningful difference between eager and lazy growth strategies:
- 2D: eager bw=0.2 flatness 0.737 vs lazy_20 bw=0.2 flatness 0.754 (within noise)
- 10D: eager bw=0.5 best_energy 0.094 vs lazy_20 bw=0.5 best_energy 0.083 (within noise)

Lazy slightly reduces final bin count (129 vs 140 for 10D bw=0.5), but the difference is negligible.

## Decision

**Default: `growth="eager"`**

Rationale:
1. Performance is equivalent to lazy across all configurations
2. Simpler implementation -- no threshold tuning needed
3. More responsive -- immediately adapts to the energy landscape
4. The slight over-expansion from outliers is not a practical problem

## Recommendations
- Default bin_width=0.2 is good for low-dimensional problems
- For higher dimensions (10D+), bin_width=0.5-1.0 is better to avoid hitting max_bins
- max_bins=200 is a reasonable default cap
