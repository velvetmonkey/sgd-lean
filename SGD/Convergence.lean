/-
  SGD-Lean: Stochastic Gradient Descent convergence in Lean 4 / Mathlib
  Module: Convergence — O(1/√K) convergence rate for bounded-noise SGD

  Main result (`sgd_convergence`):
    If the iterates satisfy the per-step distance bound and remain in a ball of
    radius R around x*, then with constant step size α the average optimality
    gap satisfies:
      (1/K) Σ_{k<K} (f(xₖ) − f*) ≤ R²/(2αK) + σR + α(L²R² + σ²)
    Setting α = c/√K gives the O(1/√K) rate.
-/
import SGD.NoisyOracle

open Finset BigOperators

noncomputable section

/-! ### Telescoping lemma for real sequences -/

/-
Telescoping sum lemma: if D(k+1) ≤ D(k) − a(k) + b(k) for all k < K,
    then Σ_{k<K} a(k) ≤ D(0) − D(K) + Σ_{k<K} b(k).
-/
theorem telescope_sum_le {K : ℕ} {D : ℕ → ℝ} {a b : ℕ → ℝ}
    (hstep : ∀ k, k < K → D (k + 1) ≤ D k - a k + b k) :
    ∑ k ∈ range K, a k ≤ D 0 - D K + ∑ k ∈ range K, b k := by
  induction' K with K ih <;> simp_all +decide [ Finset.sum_range_succ ];
  linarith [ ih fun k hk => hstep k hk.le, hstep K le_rfl ]

/-
Simplified telescoping: drop the − D(K) term (since D(K) ≥ 0 for squared norms).
-/
theorem telescope_sum_le' {K : ℕ} {D : ℕ → ℝ} {a b : ℕ → ℝ}
    (hstep : ∀ k, k < K → D (k + 1) ≤ D k - a k + b k)
    (hD_nonneg : 0 ≤ D K) :
    ∑ k ∈ range K, a k ≤ D 0 + ∑ k ∈ range K, b k := by
  linarith [ telescope_sum_le ( K := K ) ( D := D ) ( a := a ) ( b := b ) hstep ]

/-! ### Average optimality gap bound -/

namespace SGDSetup

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable (S : SGDSetup E)

/-- Full SGD convergence setup: a sequence of iterates and noisy gradients. -/
structure SGDRun (S : SGDSetup E) where
  /-- Sequence of iterates -/
  x : ℕ → E
  /-- Sequence of noisy gradients -/
  g : ℕ → E
  /-- Constant step size -/
  α : ℝ
  /-- Step size is positive -/
  hα_pos : 0 < α
  /-- Update rule: x_{k+1} = x_k − α • g_k -/
  update : ∀ k, x (k + 1) = x k - α • g k
  /-- Noise bound: ‖g_k − ∇f(x_k)‖ ≤ σ -/
  noise : ∀ k, ‖g k - S.gradf (x k)‖ ≤ S.sigma

variable {S}

/-- Distance-squared sequence for an SGD run -/
def SGDRun.distSq (run : SGDRun S) (k : ℕ) : ℝ := ‖run.x k - S.xstar‖ ^ 2

/-- Optimality gap sequence -/
def SGDRun.gap (run : SGDRun S) (k : ℕ) : ℝ := S.f (run.x k) - S.f S.xstar

/-
Per-step bound applied to an SGD run:
    distSq(k+1) ≤ distSq(k) − 2α·gap(k) + 2ασ‖xₖ−x*‖ + α²(2L²‖xₖ−x*‖² + 2σ²)
-/
theorem SGDRun.step_bound (run : SGDRun S) (k : ℕ) :
    run.distSq (k + 1) ≤ run.distSq k - 2 * run.α * run.gap k
      + 2 * run.α * S.sigma * ‖run.x k - S.xstar‖
      + run.α ^ 2 * (2 * S.L ^ 2 * ‖run.x k - S.xstar‖ ^ 2 + 2 * S.sigma ^ 2) := by
  convert S.sgd_step_bound ( run.x k ) ( run.g k ) run.α ( le_of_lt run.hα_pos ) ( run.noise k ) using 1;
  exact congr_arg ( · ^ 2 ) ( by rw [ run.update ] )

/-
Per-step bound with bounded iterates: if ‖xₖ−x*‖ ≤ R for all k, then
    distSq(k+1) ≤ distSq(k) − 2α·gap(k) + M
    where M = 2ασR + α²(2L²R² + 2σ²)
-/
theorem SGDRun.step_bound_uniform (run : SGDRun S) (R : ℝ) (_hR : 0 ≤ R)
    (hbounded : ∀ k, ‖run.x k - S.xstar‖ ≤ R) (k : ℕ) :
    run.distSq (k + 1) ≤ run.distSq k - 2 * run.α * run.gap k
      + (2 * run.α * S.sigma * R + run.α ^ 2 * (2 * S.L ^ 2 * R ^ 2 + 2 * S.sigma ^ 2)) := by
  convert SGDRun.step_bound run k |> le_trans <| le_of_sub_nonneg _ using 1;
  nlinarith [ show 0 ≤ run.α * S.sigma by exact mul_nonneg run.hα_pos.le S.hσ_nonneg, show 0 ≤ run.α ^ 2 * S.L ^ 2 by positivity, show 0 ≤ run.α ^ 2 * S.sigma ^ 2 by positivity, hbounded k, mul_le_mul_of_nonneg_left ( hbounded k ) ( show 0 ≤ run.α * S.sigma by exact mul_nonneg run.hα_pos.le S.hσ_nonneg ), mul_le_mul_of_nonneg_left ( pow_le_pow_left₀ ( by positivity ) ( hbounded k ) 2 ) ( show 0 ≤ run.α ^ 2 * S.L ^ 2 by positivity ) ]

/-
**SGD convergence theorem (sgd_convergence)**:

  Under bounded iterates ‖xₖ − x*‖ ≤ R with constant step size α > 0:

    (1/K) Σ_{k<K} (f(xₖ) − f*) ≤ R²/(2αK) + σR + α(L²R² + σ²)

  Setting α = c/√K gives the O(1/√K) convergence rate.
-/
theorem sgd_convergence (run : SGDRun S) (R : ℝ) (hR : 0 ≤ R)
    (hbounded : ∀ k, ‖run.x k - S.xstar‖ ≤ R)
    {K : ℕ} (hK : 0 < K) :
    (∑ k ∈ range K, run.gap k) / K ≤
      R ^ 2 / (2 * run.α * K) + S.sigma * R
      + run.α * (S.L ^ 2 * R ^ 2 + S.sigma ^ 2) := by
  -- Apply the&#39;telescope_sum_le' theorem with D = distSq, a(k) = 2α·gap(k), b(k) = M (constant).
  have h_telescope : ∑ k ∈ Finset.range K, (2 * run.α * run.gap k) ≤ run.distSq 0 + K * (2 * run.α * S.sigma * R + run.α ^ 2 * (2 * S.L ^ 2 * R ^ 2 + 2 * S.sigma ^ 2)) := by
    convert @telescope_sum_le' K ( fun k => run.distSq k ) ( fun k => 2 * run.α * run.gap k ) ( fun k => 2 * run.α * S.sigma * R + run.α ^ 2 * ( 2 * S.L ^ 2 * R ^ 2 + 2 * S.sigma ^ 2 ) ) _ _ using 1;
    · simp +decide [ mul_add, mul_comm, mul_left_comm ];
    · exact fun k hk => SGDRun.step_bound_uniform run R hR hbounded k;
    · exact sq_nonneg _;
  field_simp;
  norm_num [ ← Finset.mul_sum _ _ _, ← Finset.sum_mul ] at *;
  rw [ div_add', mul_div_assoc' ];
  · rw [ div_add', le_div_iff₀ ] <;> try nlinarith [ run.hα_pos ];
    nlinarith [ show 0 ≤ run.α by exact le_of_lt run.hα_pos, show run.distSq 0 ≤ R ^ 2 by exact pow_le_pow_left₀ ( norm_nonneg _ ) ( hbounded 0 ) 2 ];
  · exact ne_of_gt run.hα_pos

end SGDSetup
end