/-
  SGD-Lean: Stochastic Gradient Descent convergence in Lean 4 / Mathlib
  Module: NoisyOracle — per-step distance bound for bounded-noise SGD

  Main result (`sgd_step_bound`):
    ‖x' − x*‖² ≤ ‖x − x*‖² − 2α(f(x)−f*) + 2ασ‖x−x*‖ + α²(2L²‖x−x*‖² + 2σ²)
  where x' = x − α • g  and  ‖g − ∇f(x)‖ ≤ σ.
-/
import SGD.Defs

open InnerProductSpace

noncomputable section

namespace SGDSetup

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable (S : SGDSetup E)

/-! ### Algebraic helper: squared-norm expansion -/

/-
‖a − α • b‖² = ‖a‖² − 2α⟪b, a⟫_ℝ + α²‖b‖²
-/
theorem norm_sub_smul_sq (a b : E) (α : ℝ) :
    ‖a - α • b‖ ^ 2 = ‖a‖ ^ 2 - 2 * α * ⟪b, a⟫_ℝ + α ^ 2 * ‖b‖ ^ 2 := by
  rw [ @norm_sub_sq ℝ ];
  simp +decide [ norm_smul, mul_assoc, mul_left_comm, mul_comm, inner_smul_right ];
  rw [ real_inner_comm, mul_pow, sq_abs ]

/-! ### Inner product bound -/

/-
The inner product ⟪g, x − x*⟫ is lower-bounded using convexity and noise:
    ⟪g, x − x*⟫ ≥ (f(x) − f*) − σ ‖x − x*‖
-/
theorem inner_noisy_grad_ge (x g : E)
    (hnoise : ‖g - S.gradf x‖ ≤ S.sigma) :
    ⟪g, x - S.xstar⟫_ℝ ≥ (S.f x - S.f S.xstar) - S.sigma * ‖x - S.xstar‖ := by
  -- By Cauchy-Schwarz inequality �,� we have $|\langle g - S.gradf x, x - S.xstar \rangle| \leq \| �g� - S.gradf x\| \|x - S.xstar\|$.
  have h_cauchy_schwarz : |⟪g - S.gradf x, x - S.xstar⟫_ℝ| ≤ ‖g - S.gradf x‖ * ‖x - S.xstar‖ := by
    exact abs_real_inner_le_norm _ _;
  simp_all +decide [ abs_le, inner_sub_left ];
  nlinarith [ S.inner_grad_ge x, norm_nonneg ( x - S.xstar ) ]

/-! ### Norm squared bound -/

/-
(a + b)² ≤ 2a² + 2b² for non-negative reals
-/
theorem add_sq_le_two_mul_sq (a b : ℝ) (_ha : 0 ≤ a) (_hb : 0 ≤ b) :
    (a + b) ^ 2 ≤ 2 * a ^ 2 + 2 * b ^ 2 := by
  linarith [ sq_nonneg ( a - b ) ]

/-
‖g‖² ≤ 2L²‖x − x*‖² + 2σ²
-/
theorem noisy_grad_norm_sq_le (x g : E)
    (hnoise : ‖g - S.gradf x‖ ≤ S.sigma) :
    ‖g‖ ^ 2 ≤ 2 * S.L ^ 2 * ‖x - S.xstar‖ ^ 2 + 2 * S.sigma ^ 2 := by
  -- By the triangle inequality, ‖g‖ ≤ ‖gradf x‖ + ‖g - gradf x‖.
  have h_triangle : ‖g‖ ≤ ‖S.gradf x‖ + ‖g - S.gradf x‖ := by
    simpa using norm_add_le ( S.gradf x ) ( g - S.gradf x );
  refine' le_trans ( pow_le_pow_left₀ ( norm_nonneg _ ) h_triangle 2 ) _;
  exact le_trans ( pow_le_pow_left₀ ( by positivity ) ( add_le_add ( S.grad_norm_le x ) hnoise ) 2 ) ( by nlinarith [ add_sq_le_two_mul_sq ( S.L * ‖x - S.xstar‖ ) S.sigma ( mul_nonneg S.hL_nonneg ( norm_nonneg _ ) ) S.hσ_nonneg ] )

/-! ### Main per-step bound -/

/-
**Per-step distance bound (sgd_step_bound)**.

  If `x' = x − α • g` with `‖g − ∇f(x)‖ ≤ σ` and `α ≥ 0`, then:

  `‖x' − x*‖² ≤ ‖x − x*‖² − 2α(f(x) − f*) + 2ασ‖x − x*‖ + α²(2L²‖x − x*‖² + 2σ²)`
-/
theorem sgd_step_bound (x g : E) (α : ℝ) (hα : 0 ≤ α)
    (hnoise : ‖g - S.gradf x‖ ≤ S.sigma) :
    ‖(x - α • g) - S.xstar‖ ^ 2 ≤
      ‖x - S.xstar‖ ^ 2 - 2 * α * (S.f x - S.f S.xstar)
      + 2 * α * S.sigma * ‖x - S.xstar‖
      + α ^ 2 * (2 * S.L ^ 2 * ‖x - S.xstar‖ ^ 2 + 2 * S.sigma ^ 2) := by
  -- Apply the norm_sub_smul_sq theorem to expand(x - α • g) - S.xstar‖².
  have h_expand : ‖(x - α • g) - S.xstar‖ ^ 2 = ‖x - S.xstar‖ ^ 2 - 2 * α * ⟪g, x - S.xstar⟫_ℝ + α ^ 2 * ‖g‖ ^ 2 := by
    convert norm_sub_smul_sq ( x - S.xstar ) g α using 1 ; abel_nf;
  nlinarith [ inner_noisy_grad_ge S x g hnoise, noisy_grad_norm_sq_le S x g hnoise ]

end SGDSetup
end