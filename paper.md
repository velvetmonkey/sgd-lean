# sgd-lean: Formal Proofs of Bounded-Noise SGD Convergence in Lean 4

Ben Cassie  
2026

## Abstract

`sgd-lean` is a Lean 4 / Mathlib library formalising a bounded-noise convergence theorem for stochastic gradient descent. The library works over a real inner product space, packages a convex and smooth objective in an `SGDSetup` structure, models noisy gradients by a deterministic pointwise bound, proves a one-step distance inequality for the SGD update, and derives the standard averaged `O(1/sqrt K)` rate up to an irreducible bounded-noise term. The development contains zero `sorry`, zero `admit`, and uses standard Lean/Mathlib axioms only. Its contribution is a machine-checked reference for the core algebra behind noisy first-order optimisation, stated in a form that can be imported by future formal work on learning dynamics, robustness, and AI safety.

## 1. Introduction

Stochastic gradient descent is one of the central algorithms of modern machine learning. Its attraction is practical: it replaces exact gradients by cheaper, noisier gradient estimates and still gives useful convergence guarantees. The informal theorem is familiar. For convex objectives, suitably controlled gradient noise gives an averaged optimality-gap rate of order `O(1/sqrt K)`.

The fully stochastic version of this theorem is usually stated in expectation and depends on filtrations, conditional expectations, martingale differences, and variance bounds. Those ingredients are not yet as convenient in Mathlib as the deterministic norm and inner-product tools used throughout first-order optimisation. `sgd-lean` therefore formalises a deterministic bounded-noise surrogate:

```text
||g_k - grad f(x_k)|| <= sigma.
```

This is not a claim that all stochastic analysis is deterministic. It is a carefully scoped theorem: if every oracle vector is within `sigma` of the true gradient, then the same SGD algebra gives a quantitative averaged bound. The price is an additive `sigma R` term, where `R` bounds the distance from the iterates to the minimiser. This term is the worst-case noise floor. In an in-expectation theorem it would normally be replaced by a variance contribution that can average away under stronger assumptions.

The formal setting is a real inner product space `E`. An `SGDSetup` packages an objective `f : E -> R`, an abstract gradient map `gradf : E -> E`, a minimiser `xstar`, a Lipschitz constant `L`, and a noise bound `sigma`. The update is the standard first-order step

```text
x' = x - alpha • g.
```

The library proves that if the iterates remain in a ball of radius `R` around `xstar`, then

```text
(1/K) * sum_{k<K} (f(x_k) - f*) <=
  R^2 / (2 alpha K) + sigma R + alpha (L^2 R^2 + sigma^2).
```

Choosing `alpha = c / sqrt K` balances the first and third terms and gives the expected `O(1/sqrt K)` dependence, with the deterministic bounded-noise floor still visible.

## 2. Library Overview

The project is organised into three implementation modules plus a root import file:

- `SGD/Defs.lean` defines `SGDSetup`, first-order convexity, L-smoothness through an abstract gradient map, the gradient norm bound, non-negativity of the optimality gap, and `sgd_step`.
- `SGD/NoisyOracle.lean` proves the algebraic and analytic estimates that lead to the one-step bounded-noise distance bound.
- `SGD/Convergence.lean` proves the telescoping lemmas, defines the `SGDRun` trajectory structure, and derives the averaged `O(1/sqrt K)` convergence statement.
- `SGD.lean` is the root module importing the library.

The project depends on Lean `v4.28.0` and Mathlib `v4.28.0`.

The `SGDSetup` structure deliberately uses an abstract gradient map rather than extracting a Frechet derivative from `f`. Convexity and smoothness are stated as first-order hypotheses about `f` and `gradf`. This keeps the proof focused on the convergence argument rather than differentiability infrastructure.

The main trajectory structure, `SGDRun`, stores the iterate sequence, noisy-gradient sequence, constant step size, update rule, and noise bound. The convergence theorem is then stated for any such run whose iterates are uniformly bounded around the minimiser.

## 3. Theorem Inventory

The source contains eight headline results, organised into three layers.

### Layer 1 - Oracle and Step Mechanics

1. `norm_sub_smul_sq` — The exact squared-norm identity for a scaled step:

```text
||a - alpha • b||^2 =
  ||a||^2 - 2 alpha <b, a> + alpha^2 ||b||^2.
```

This is the algebraic starting point for the per-step analysis.

2. `inner_noisy_grad_ge` — The noisy gradient still has a useful inner-product lower bound:

```text
<g, x - x*> >= (f(x) - f*) - sigma ||x - x*||.
```

The proof combines first-order convexity for the true gradient with Cauchy-Schwarz on the noise vector `g - gradf x`.

3. `noisy_grad_norm_sq_le` — The squared norm of the noisy gradient is bounded by the smoothness scale and the noise:

```text
||g||^2 <= 2 L^2 ||x - x*||^2 + 2 sigma^2.
```

This is obtained from the triangle inequality, the Lipschitz-gradient bound, and `(a+b)^2 <= 2a^2 + 2b^2`.

4. `sgd_step_bound` — The core per-step distance estimate:

```text
||x' - x*||^2 <= ||x - x*||^2
  - 2 alpha (f(x) - f*)
  + 2 alpha sigma ||x - x*||
  + alpha^2 (2 L^2 ||x - x*||^2 + 2 sigma^2).
```

This theorem is the local SGD argument in its reusable form.

### Layer 2 - Telescoping Infrastructure

5. `telescope_sum_le` — A general telescoping lemma:

```text
D(k+1) <= D(k) - a(k) + b(k)
```

implies

```text
sum a(k) <= D(0) - D(K) + sum b(k).
```

6. `telescope_sum_le'` — A variant that drops the terminal term when `D(K) >= 0`:

```text
sum a(k) <= D(0) + sum b(k).
```

For SGD, `D(k)` is the squared distance to `xstar`, so this non-negativity condition is automatic.

### Layer 3 - Convergence

7. `SGDRun.step_bound_uniform` — The per-step bound specialised to a run whose iterates satisfy

```text
||x_k - x*|| <= R.
```

The result replaces all distance-dependent error terms by constants involving `R`.

8. `sgd_convergence` — The averaged bounded-noise convergence theorem:

```text
(1/K) * sum_{k<K} (f(x_k) - f*) <=
  R^2 / (2 alpha K) + sigma R + alpha (L^2 R^2 + sigma^2).
```

Setting `alpha = c / sqrt K` gives the advertised `O(1/sqrt K)` rate, with the additive `sigma R` worst-case noise floor.

## 4. Key Technical Highlights

### Bounded Noise Rather Than Full Stochasticity

The library uses a deterministic bounded-noise oracle because the fully stochastic theorem requires probability infrastructure that is outside the immediate optimisation core. In the usual stochastic analysis, one asks that `g_k` be unbiased or have bounded variance relative to the past. In Lean this means handling filtrations, conditional expectation, and adapted processes.

The bounded-noise model isolates the part of the proof that is purely Hilbert-space geometry. The assumption `||g_k - gradf(x_k)|| <= sigma` is strong, but it has a clear interpretation: every observed gradient is within a fixed ball around the true gradient. The convergence theorem therefore includes the term `sigma R`. If adversarial bounded noise is always allowed, no argument can force the average gap below the scale at which noise can consistently point against progress.

### The Per-Step Bound

The theorem `sgd_step_bound` is where all pieces meet. The norm identity expands the next squared distance. Convexity turns the true-gradient inner product into the objective gap. Cauchy-Schwarz controls the error introduced by replacing the true gradient with `g`. The noisy-gradient norm bound controls the `alpha^2 ||g||^2` term.

This decomposition is the formal version of the standard SGD calculation. Lean forces each inequality to be stated with the correct sign and with the non-negativity hypotheses needed for multiplication and squaring.

### Telescoping

After applying the uniform radius bound, the per-step theorem has the form

```text
D(k+1) <= D(k) - 2 alpha gap(k) + M.
```

Summing over `k < K` cancels the adjacent distance terms. The terminal squared distance is non-negative, so it can be discarded. What remains is a bound on the sum of the gaps, and division by `K` gives the average.

### The `O(1/sqrt K)` Step Size

The final bound has three terms. The first decreases like `1/(alpha K)`, the last grows like `alpha`, and the middle `sigma R` term is independent of the step size. Choosing `alpha = c / sqrt K` balances the decreasing and increasing terms. This is the same rate scale as subgradient methods and mirror descent, reflecting that bounded noise destroys the faster deterministic `O(1/k)` behaviour available for smooth gradient descent.

## 5. Relation to Sibling Libraries

`gradient-descent-lean` formalises deterministic smooth convex gradient descent and has DOI `10.5281/zenodo.20472996`. Its deterministic setting supports an `O(1/k)` rate. `sgd-lean` adds bounded noise and pays the expected `O(1/sqrt K)` cost, plus the deterministic noise floor.

`nesterov-lean` has DOI `10.5281/zenodo.20474481` and proves accelerated deterministic convergence with an `O(1/k^2)` rate. `sgd-lean` is not an acceleration theorem; it is the noisy first-order counterpart where gradient observations are imperfect.

`subgradient-lean` proves the same `O(1/sqrt K)` rate for non-smooth deterministic convex objectives. The two libraries have related telescoping structures but different reasons for the slower rate: non-smoothness in one case, bounded gradient noise in the other.

`mirror-descent-lean` gives the Bregman-geometric generalisation of the same template. Its proof uses a mirror map and Bregman divergence, whereas `sgd-lean` stays in Euclidean Hilbert geometry.

## 6. AI Safety Significance

Modern AI systems are built from optimisation processes whose gradients are noisy, approximate, minibatched, clipped, quantised, or otherwise imperfect. Safety arguments that treat optimisation as exact descent can miss the effects of persistent error. A formal bounded-noise theorem makes the price of imperfect gradients explicit.

The additive `sigma R` term is especially important. It says that bounded but adversarially directed noise can prevent convergence to the exact optimum. That is a useful lesson for safety work: robustness claims should specify whether noise averages away, is unbiased, or can persist in a harmful direction.

The library does not certify any deployed training run. It supplies a verified component that can be reused in larger formal arguments about noisy learning dynamics, stability under approximate computation, and the gap between idealised optimisation and implemented algorithms.

## 7. Conclusion

`sgd-lean` provides a focused Lean 4 formalisation of bounded-noise SGD convergence. It defines the optimisation setup, the deterministic noisy oracle model, the SGD update, the per-step distance bound, telescoping infrastructure, and the averaged convergence theorem. The result is a checked account of the standard `O(1/sqrt K)` SGD proof in a form that avoids probabilistic infrastructure while preserving the core geometry of noisy first-order optimisation.

## References

Robbins, H. and Monro, S. (1951). *A stochastic approximation method*. Annals of Mathematical Statistics, 22(3), 400-407.

Shalev-Shwartz, S. and Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge University Press.

The Mathlib Community. (2024). *The Lean Mathematical Library*. GitHub repository. <https://github.com/leanprover-community/mathlib4>

Cassie, B. (2026). *gradient-descent-lean: Formal Proofs of Gradient Descent Convergence in Lean 4*. Zenodo. <https://doi.org/10.5281/zenodo.20472996>

Cassie, B. (2026). *nesterov-lean: Formal Proofs of Nesterov Accelerated Gradient Descent in Lean 4*. Zenodo. <https://doi.org/10.5281/zenodo.20474481>

