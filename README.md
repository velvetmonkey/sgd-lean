# sgd-lean

[![Lean 4](https://img.shields.io/badge/Lean-4.28.0-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-v4.28.0-purple)](https://github.com/leanprover-community/mathlib4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Proofs](https://img.shields.io/badge/proofs-proven%20%2F%200%20sorry-brightgreen)](SGD)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20475583-blue)](https://zenodo.org/records/20475583)

**sgd-lean: Formal Proofs of Bounded-Noise SGD Convergence in Lean 4**

Lean 4 formal proofs of stochastic gradient descent convergence under a deterministic bounded-noise oracle: a per-step distance bound and an O(1/√K) averaged-regret rate.

**Zero sorry statements.** Standard axioms only (`propext`, `Classical.choice`, `Quot.sound`).

## What this is, and why it matters

The headline theorem is `sgd_convergence` in `SGD/Convergence.lean`. For a positive constant step size, a positive horizon, and iterates bounded by radius `R` around the designated minimizer, it bounds the average objective gap by `R² / (2αK) + σR + α(L²R² + σ²)`.

The proof first establishes a one-step squared-distance inequality for the noisy update. The bounded-iterate hypothesis turns the noise and gradient terms into a uniform remainder, after which a general telescoping lemma sums the inequalities and the initial distance is bounded by `R²`.

The oracle model is deterministic: each supplied gradient is within `σ` of an abstract gradient map. Consequently, this is not an in-expectation stochastic theorem, and the bound retains the worst-case floor `σR`. Convexity and smoothness are encoded as first-order assumptions, and boundedness of every iterate is also assumed explicitly.

## Background and motivation

Stochastic gradient descent is the workhorse of modern machine learning, but its iterates use *noisy* gradients rather than the true gradient. The central question is whether that noise still allows convergence — and at what rate. For convex objectives the answer is the classic **O(1/√K)** rate on the averaged optimality gap.

This library machine-checks that rate in Lean 4. To stay within Mathlib's current capabilities, it uses a **deterministic bounded-noise model** rather than a fully stochastic one: each oracle gradient gₖ is only assumed to satisfy ‖gₖ − ∇f(xₖ)‖ ≤ σ. Mathlib lacks the conditional-expectation / filtration infrastructure needed to state and manipulate the stochastic (in-expectation) version, so the bounded-noise model is the faithful deterministic surrogate. It yields an O(1/√K) + O(σR) bound — the σR term is the irreducible worst-case-noise error that would vanish in expectation in the stochastic setting.

## Setting

A real inner product space `E`, with a convex, L-smooth objective f : E → ℝ, minimiser x*, and an abstract gradient map ∇f stated via first-order conditions (avoiding heavy differentiability infrastructure). The oracle is **deterministic bounded-noise**:

```
‖g_k − ∇f(x_k)‖ ≤ σ
```

The update is `x_{k+1} = x_k − α • g_k` with constant step size α > 0.

## Main result

Under bounded iterates ‖xₖ − x*‖ ≤ R with constant step α > 0:

```
(1/K) Σ_{k<K} (f(x_k) − f*) ≤ R²/(2αK) + σR + α(L²R² + σ²)
```

Setting α = c/√K balances the R²/(2αK) and α(L²R² + σ²) terms, giving the **O(1/√K)** rate (with the additive σR worst-case-noise floor).

## Project structure

```
SGD/
├── Defs.lean         — SGDSetup (convex L-smooth f, bounded-noise σ), first-order
│                       conditions, gap_nonneg, sgd_step
├── NoisyOracle.lean  — per-step distance bound: norm_sub_smul_sq,
│                       inner_noisy_grad_ge, noisy_grad_norm_sq_le, sgd_step_bound
└── Convergence.lean  — telescoping lemmas, SGDRun trajectory, sgd_convergence
SGD.lean              — Root module
```

## Theorem inventory

8 headline results.

| # | Name | Statement |
|---|------|-----------|
| 1 | `norm_sub_smul_sq` | ‖a − α•b‖² = ‖a‖² − 2α⟪b,a⟫ + α²‖b‖² (squared-norm expansion) |
| 2 | `inner_noisy_grad_ge` | ⟪g, x−x*⟫ ≥ (f(x)−f*) − σ‖x−x*‖ (convexity + Cauchy–Schwarz) |
| 3 | `noisy_grad_norm_sq_le` | ‖g‖² ≤ 2L²‖x−x*‖² + 2σ² (noisy gradient norm bound) |
| 4 | `sgd_step_bound` | ‖x′−x*‖² ≤ ‖x−x*‖² − 2α(f(x)−f*) + 2ασ‖x−x*‖ + α²(2L²‖x−x*‖²+2σ²) |
| 5 | `telescope_sum_le` | Σ aₖ ≤ D(0) − D(K) + Σ bₖ when D(k+1) ≤ D(k) − aₖ + bₖ |
| 6 | `telescope_sum_le'` | Σ aₖ ≤ D(0) + Σ bₖ (dropping − D(K) ≥ 0) |
| 7 | `SGDRun.step_bound_uniform` | per-step bound with bounded iterates ‖xₖ−x*‖ ≤ R |
| 8 | `sgd_convergence` | (1/K) Σ (f(xₖ)−f*) ≤ R²/(2αK) + σR + α(L²R²+σ²) — O(1/√K) |

## Design note

The analysis uses a deterministic bounded-noise oracle (‖gₖ − ∇f(xₖ)‖ ≤ σ) rather than a stochastic one, because Mathlib does not yet provide the conditional-expectation / filtration machinery required for the in-expectation analysis. The objective's convexity and L-smoothness are stated as first-order conditions on an abstract gradient map, keeping the development self-contained and free of Fréchet-derivative bookkeeping while remaining valid over a general real inner product space.

## Dependencies

- Lean 4.28.0
- Mathlib v4.28.0

## Paper

**sgd-lean: Formal Proofs of Bounded-Noise SGD Convergence in Lean 4**
Ben Cassie (2026). Companion paper: [paper.md](paper.md).

DOI: https://doi.org/10.5281/zenodo.20475583

## Related work

- [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) — Lean 4 gradient descent convergence (O(1/k) rate)
- [nesterov-lean](https://github.com/velvetmonkey/nesterov-lean) — Lean 4 Nesterov accelerated gradient descent (O(1/k²) rate)
- [subgradient-lean](https://github.com/velvetmonkey/subgradient-lean) — Lean 4 subgradient method convergence (O(1/√k) rate)
- [mirror-descent-lean](https://github.com/velvetmonkey/mirror-descent-lean) — Lean 4 mirror descent with Bregman divergences (O(1/√K) rate)

## Acknowledgements

Proofs in this library were generated using [Aristotle](https://aristotle.harmonic.fun), an AI proof assistant for Lean 4 and Mathlib. The proof discipline — zero sorry, standard axioms only — was specified by the author and enforced by the Lean type checker.

## Author

Ben Cassie · [@thevelvetmonke](https://x.com/thevelvetmonke)
## Part of the Lean proof corpus

One of a family of small, machine-checked Lean 4 developments. Index: [velvetmonkey/lean](https://github.com/velvetmonkey/lean) ([live index](https://velvetmonkey.github.io/lean)).
