/-
  SGD-Lean: Stochastic Gradient Descent convergence in Lean 4 / Mathlib
  Module: Defs — core definitions for the bounded-noise SGD model

  Setting:
    E  — a real Hilbert space
    f  : E → ℝ  — convex, L-smooth objective
    Bounded-noise oracle: gₖ satisfies ‖gₖ − ∇f(xₖ)‖ ≤ σ
-/
import Mathlib

open InnerProductSpace

noncomputable section

/-! ### Core definitions -/

/-- Configuration for the bounded-noise SGD analysis.

  We work with an abstract "gradient map" `gradf : E → E` rather than extracting the
  Fréchet derivative, and state the convexity / smoothness properties as first-order
  conditions on `gradf` and `f`. This keeps the analysis self-contained and avoids
  heavy differentiability infrastructure. -/
structure SGDSetup (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  /-- Objective function -/
  f : E → ℝ
  /-- Gradient map (abstract; need not literally be the Fréchet derivative) -/
  gradf : E → E
  /-- Minimiser -/
  xstar : E
  /-- Lipschitz constant of the gradient (L ≥ 0) -/
  L : ℝ
  /-- Noise bound (σ ≥ 0) -/
  sigma : ℝ
  /-- L is non-negative -/
  hL_nonneg : 0 ≤ L
  /-- σ is non-negative -/
  hσ_nonneg : 0 ≤ sigma
  /-- First-order convexity condition (for all x, y):
      f(y) ≥ f(x) + ⟪∇f(x), y − x⟫  -/
  convexity : ∀ x y : E, ⟪gradf x, y - x⟫_ℝ ≤ f y - f x
  /-- Gradient is zero at the minimiser -/
  grad_min : gradf xstar = 0
  /-- L-smoothness (Lipschitz gradient): ‖∇f(x) − ∇f(y)‖ ≤ L ‖x − y‖ -/
  smooth : ∀ x y : E, ‖gradf x - gradf y‖ ≤ L * ‖x - y‖

namespace SGDSetup

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable (S : SGDSetup E)

/-- The gradient convexity condition in the form used for descent:
    ⟪∇f(x), x − x*⟫ ≥ f(x) − f(x*) -/
theorem inner_grad_ge (x : E) : ⟪S.gradf x, x - S.xstar⟫_ℝ ≥ S.f x - S.f S.xstar := by
  have h := S.convexity x S.xstar
  have : ⟪S.gradf x, S.xstar - x⟫_ℝ = -⟪S.gradf x, x - S.xstar⟫_ℝ := by
    rw [← neg_sub, inner_neg_right]
  linarith [this]

/-- Gradient norm bound at any point: ‖∇f(x)‖ ≤ L ‖x − x*‖ -/
theorem grad_norm_le (x : E) : ‖S.gradf x‖ ≤ S.L * ‖x - S.xstar‖ := by
  have h := S.smooth x S.xstar
  rwa [S.grad_min, sub_zero] at h

/-- f(x) ≥ f(x*) for all x (consequence of convexity + gradient zero at minimum) -/
theorem f_ge_fstar (x : E) : S.f x ≥ S.f S.xstar := by
  have h := S.convexity S.xstar x
  simp [S.grad_min, inner_zero_left] at h
  linarith

/-- The optimality gap is non-negative -/
theorem gap_nonneg (x : E) : 0 ≤ S.f x - S.f S.xstar := by linarith [S.f_ge_fstar x]

/-- One step of noisy/bounded-noise SGD: x' = x − α • g -/
def sgd_step (x g : E) (α : ℝ) : E := x - α • g

end SGDSetup
end
