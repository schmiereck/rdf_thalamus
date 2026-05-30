# RDF Milestone Review — Iteration 029 — Null Result: M2 (SFA+VICReg) Does Not Clear Practical-Significance Threshold on Separate Backbone

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (verbatim from the iter_029 plan): "Adding an
explicit SFA slowness term (||z_dyn(t) - z_dyn(t-1)||²) to the
VICReg-only z_dyn objective on the separate-backbone architecture
will improve identity encoding such that mean ΔR²_color ≥ 0.30 across
the union seed bank, with collapse rate ≤ 10% and no centroid_MSE
degradation beyond 110."

Falsification criteria:
- F1: Arm B (SFA+VICReg, separate backbone) mean ΔR²_color < 0.30 →
  hypothesis falsified.
- F2: collapse rate > 10% on any arm → power criterion failed.
- F3: centroid_MSE > 110 → spatial readout degraded.

## 2. Experimental Protocol
- Architecture: `NonParametricJEPASpatial` with separate z_coord and
  z_dyn backbones (iter_027 Arm C topology), d_max = 16, d_t frozen at 3.
- Arm A (control): VICReg-only z_dyn (mask_dyn_sim=True), JEPA mode,
  coord_vicreg=True.
- Arm B (test): SFA + VICReg on z_dyn, SFA weight = 1.0, coord_vicreg=True.
- Arm B′ (perturbation, included for Gate 3): SFA + VICReg with SFA
  weight = 5.0.
- Seed bank: union of original (including hard seeds 53, 71) and fresh
  banks, n = 30 per arm (60 runs total reported as completing without
  timeout).
- Buffer: 4000 (held constant from iter_026).
- Training step count and exact hyperparameters: per iter_029
  pre-registration; not all values re-stated here.
- Metrics: ΔR²_color (primary), mean_abs_corr, centroid_MSE,
  train + eval std (collapse gate).

## 3. Observed Quantities
- Arm A (VICReg-only, control): mean ΔR²_color = 0.0445.
- Arm B (SFA+VICReg, weight 1.0): mean ΔR²_color = 0.2749, σ = 0.577
  across 30 seeds.
  - Original seeds subset: mean 0.1921.
  - Fresh seeds subset: mean 0.3576.
  - Hard seeds (53, 71): no improvement over Arm A.
- Arm B′ (SFA weight 5.0): lower mean than Arm B (i.e. higher SFA
  weight made the result worse, not better).
- SFA loss decreased during training (mean final = 0.1408),
  confirming the slowness objective was active and the optimization
  converged.
- Collapse rate: 0% across all 60 runs (F2 passed).
- centroid_MSE: within the F3 envelope (F3 passed).
- Resolution note: ΔR²_color is dimensionless in [-∞, 1]; 0.30 was
  chosen as the practical-significance threshold ex ante.

## 4. Verdict
**Refuted (on the pre-declared primary criterion).** Arm B mean
ΔR²_color = 0.2749 < 0.30; F1 triggered.

Two important qualifications:
- The result is **consistent with a directional improvement** of SFA
  over VICReg-only (6.2× ratio in mean), but the high run-to-run
  variance (σ = 0.577) and the failure of higher SFA weights to
  monotonically improve the result mean it does not pass Gate 3
  (Parameter-Tuning Hygiene). Per the project's reporting standards,
  this is "suggestive evidence at best."
- The post-hoc subsetting by seed bank (fresh seeds 0.3576 clears the
  threshold; original seeds 0.1921 does not) is **not** a basis for
  re-declaring success. The union seed bank is the pre-registered
  population; subsetting to the part that clears the threshold is
  selection bias.

Combined with iter_023–024 (SFA refuted on shared backbone), the
evidence now spans both architectural regimes and consistently shows
that explicit slowness does not reliably produce identity encoding
above the practical-significance threshold in this task.

## 5. Construction-vs-Empirical Note
The 0% collapse rate across all 60 runs is partly construction-driven:
VICReg's variance hinge directly enforces per-dimension std ≥ 1, which
is the same quantity used in the collapse gate. So the absence of
collapse, under any configuration that includes pooled VICReg, is
expected by the chosen objective rather than an empirical discovery
about the encoder's dynamics.

The empirically meaningful quantity in this iteration is **ΔR²_color**,
which measures whether downstream color identity can be linearly
decoded from z_dyn. This is not a quantity enforced by the loss
function and is therefore genuinely informative. The result that
ΔR²_color did not reach the pre-declared threshold under the
hypothesized objective is a clean empirical null about the
*representational content* of z_dyn, not about its variance.

The SFA term itself is an empirical objective, not a construction
identity — slowness on z_dyn does not by construction guarantee
identity encoding (it could in principle satisfy slowness via a
constant-up-to-VICReg-variance representation that carries no color
information). The iter_029 result is consistent with this latter
failure mode: SFA loss decreased while identity decoding only partly
improved.

## 6. Limitations
- This result does not establish that SFA is the wrong objective in
  principle; it establishes that SFA + VICReg as specified by M2,
  on the separate-backbone architecture, with the chosen weights and
  seed bank, did not clear the pre-declared identity-encoding
  threshold reliably.
- It does not rule out that SFA can clear the threshold under a
  variance-ramped or weight-annealed protocol in which the VICReg
  variance hinge is not competing with slowness throughout training.
  iter_030 D2 will test this.
- It does not establish that the SFA mandate (M2) should be discarded
  project-wide; the cross-architecture convergence of the null is
  grounds to challenge it, not to overturn it. Discarding M2 requires
  that an alternative identity objective (e.g. contrastive identity
  binding, iter_030 D3) clear the same threshold under the same
  population, demonstrating that the failure was in slowness as the
  organizing prior, not in the task or evaluation.
- The σ = 0.577 variance on the primary metric means the M2 result is
  not just below threshold in mean — it is unstable enough that no
  single-seed run can be interpreted as evidence. Future
  representation-objective comparisons in this project must include
  a variance-stability gate, not only a mean gate.
- Hard seeds 53 and 71 continue to function as a stress test that no
  objective tested so far has cleared. Their failure mode is not yet
  mechanistically characterized.