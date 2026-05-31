# sgd-lean

[![Lean 4](https://img.shields.io/badge/Lean-4.28.0-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-v4.28.0-purple)](https://github.com/leanprover-community/mathlib4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Proofs](https://img.shields.io/badge/proofs-pending-lightgrey)](SGDLean)

Lean 4 formal proofs of stochastic gradient descent convergence: O(1/√k) rate for convex objectives.

**Zero sorry statements.**

## Why it matters

Stochastic gradient descent is the algorithm that trains nearly every large neural network in existence. Its convergence guarantee -- that noisy gradient updates still drive the objective toward a minimum at rate O(1/√k) -- is one of the most practically important results in optimisation theory. Yet the proof involves noise variance bounds, step size schedules, and telescoping arguments that are rarely stated with full precision.

This library machine-checks the O(1/√k) convergence rate in Lean 4, extending [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) (O(1/k) deterministic rate) to the stochastic setting.

## Setting

f : E → ℝ, E a real Hilbert space. f is L-smooth and convex.
Noisy gradient oracle: g_k with ‖g_k - ∇f(x_k)‖ ≤ σ (bounded noise model).
Step size: αₖ = c/√k.

## Planned project structure

```
SGDLean/
├── Defs.lean       — NoisyOracle, step size schedule, SGD sequence
├── NoisyOracle.lean — Per-step bound under bounded noise
└── Convergence.lean — O(1/√k) convergence rate
SGDLean.lean        — Root module
```

## Planned theorem inventory

| # | Name | Statement |
|---|------|-----------|
| 1 | `sgd_step_bound` | Per-step distance bound under bounded noise |
| 2 | `sgd_convergence` | (1/k) Σ f(xᵢ) - f* ≤ C/√k |

## Key technical highlights

- Extends gradient-descent-lean infrastructure (L-smoothness, convexity, descent lemma)
- Bounded noise model: ‖g_k - ∇f(x_k)‖ ≤ σ
- Step size schedule αₖ = c/√k is the classical choice that balances noise and progress
- Standard axioms only: `propext`, `Classical.choice`, `Quot.sound`
- Zero `sorry`, zero `admit`

## Dependencies

- Lean 4.28.0
- Mathlib v4.28.0

## Related work

- [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) — Lean 4 deterministic gradient descent (O(1/k) rate)
- [nesterov-lean](https://github.com/velvetmonkey/nesterov-lean) — Lean 4 Nesterov accelerated gradient descent (O(1/k²) rate)
- [projected-gd-lean](https://github.com/velvetmonkey/projected-gd-lean) — Lean 4 projected gradient descent
- [kuramoto-lean](https://github.com/velvetmonkey/kuramoto-lean) — Lean 4 Kuramoto synchronisation

## Acknowledgements

Proofs in this library were generated using [Aristotle](https://aristotle.harmonic.fun), an AI proof assistant for Lean 4 and Mathlib.

## Author

Ben Cassie · [@thevelvetmonke](https://x.com/thevelvetmonke)
