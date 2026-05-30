# RDF Milestone Review — Iteration 032 — Null Result: Centroid-Gated Readout Architectural Fix Falsified, Project Pivots to Behavioral Evaluation by Pre-Committed Rule

## 1. Pre-Declared Hypothesis and Falsification Criterion
Verbatim from iter_032 pre-registration:

> "Replacing mean-pool z_dyn with attention-pooled multi-dimensional
> feature vectors gated by the coord backbone's soft-argmax attention
> achieves ΔR²_color ≥ 0.30 with variance stability."

Binding decision rule (pre-committed before the run):
> "(a) If the rich readout clears ΔR²_color ≥ 0.30 with lower CI ≥
> 0.18 and collapse ≤ 10%, the representation is solved — advance to
> Phase 2/3 integration. (b) If it yields only another partial gain,
> that is the third convergent signal (after iter_021 CGIR and the
> 5-objective null) that ΔR²_color ≥ 0.30 is the wrong target, and
> the project hard-pivots to behavioral evaluation."

Falsification criterion for the hypothesis itself: any of {ΔR²_color
< 0.30, lower CI < 0.18, collapse > 10%} on the E2 (K=4) arm.

## 2. Experimental Protocol
- Substrate: existing `NonParametricJEPASpatial` separate-backbone CNN
  (iter_028 config: mask_dyn_sim, coord_vicreg).
- Three arms:
  - E1: Mean-pool readout + VICReg-only (control / current best).
  - E1.5: Scalar centroid-gated readout (K=1) + VICReg-only.
  - E2: Rich attention-pooled centroid-gated readout (K=4 multi-
    dimensional feature vector) + VICReg-only.
- Frozen: d_t=3, decoder-free, no positional encoding, buffer=4000,
  seeds drawn from union seed bank including hard seeds 53, 71.
- Collapse criterion: existing `has_collapsed` plus per-dimension std
  < 0.5 (the M1 variance-hinge threshold).
- Metric: ΔR²_color (per-object color regression delta over a
  null-model baseline), measured on a held-out evaluation buffer.

## 3. Observed Quantities
- E2 (rich, K=4): **100% collapse**, per_dim_std ~0.3–0.6 (below 0.5
  viability threshold). ΔR²_color undefined / not computed on
  collapsed seeds.
- E1.5 (scalar centroid, K=1): **10% collapse**, ΔR²_color worse than
  E1 mean-pool on non-collapsed seeds (degraded relative to control).
- E1 (mean-pool control): 0% collapse, ΔR²_color ~0.045 (consistent
  with iter_029 baseline).
- Falsification threshold (ΔR²_color ≥ 0.30, lower CI ≥ 0.18,
  collapse ≤ 10%): violated on all three counts for E2; collapse
  threshold violated and ΔR²_color degraded for E1.5.

## 4. Verdict
**Refuted.** Both K=1 and K=4 variants of the centroid-gated readout
failed the pre-registered gate. The K=4 arm failed by a new mechanism
(cross-backbone attention coupling under VICReg → catastrophic
collapse), and the K=1 arm failed by producing worse identity
encoding than mean-pool while introducing collapse. The pre-committed
binding decision rule triggers branch (b): hard-pivot to behavioral
evaluation.

This is the project's *third* convergent signal that ΔR²_color ≥ 0.30
is not achievable representation-side under the current frozen-
constraint set (decoder-free + mean-pool readout family + fixed
dimensionality):
1. iter_021 CGIR — partial gain (+0.124), missed 0.30.
2. iter_023–031 — 5-objective convergent null (SFA, JEPA, temporal-
   contrastive, variance-ramped SFA, reconstruction).
3. iter_032 — readout-architecture null (K=1 worse, K=4 collapse).

## 5. Construction-vs-Empirical Note
Genuinely empirical: the cross-backbone attention coupling collapse
was not predicted from construction. The pre-registration anticipated
one of two outcomes (clear 0.30, or partial gain like CGIR). What
actually happened — *worse* than mean-pool plus catastrophic
collapse — was a third outcome with a new mechanistic story
(peaked softmax concentrates VICReg variance constraint at the
attended spatial position, driving degeneracy). The K=4-worse-than-
K=1 ordering further confirms this is not a construction artifact:
if it were, the higher-dimensional readout should not collapse
*more*, since K=4 carries strictly more capacity than K=1.

Note: the *binding pivot decision* in branch (b) is not itself an
empirical claim — it is the execution of a pre-committed protocol
rule. Its scientific status is "we did what we said we would do
before seeing the data."

## 6. Limitations
- This result does **not** show that ΔR²_color ≥ 0.30 is *unachievable*
  on any architecture. It shows the three categorical interventions
  tried so far (objective swap, readout fix, supervised reconstruction)
  do not reach it under the project's frozen constraints. A
  constraint-relaxation step (decoder, different readout family, or
  higher d_t) is the next architectural variable, and is reserved for
  iter_034+ conditional on the iter_033 behavioral pivot outcome.
- The pivot to behavioral evaluation is **not** evidence that the
  representation is "good enough for behavior." It is the execution
  of a pre-committed rule that says the question is now worth asking
  directly. iter_033 will measure whether the answer is yes or no
  against pre-registered, control-constructed thresholds.
- The 5-objective convergent null (iter_023–031) was measured under
  the now-known-broken mean-pool readout. It is not yet a clean
  falsification of the M2 mandate at the objective level, because a
  fair re-test would require a working non-mean-pool readout, and
  iter_032's attempted readout fix failed. M2's empirical status
  remains "untestable on the current architecture" rather than
  "falsified."
- The iter_032 cross-backbone coupling collapse is mechanistically
  plausible but has not been independently confirmed by an ablation
  (e.g., the same K=4 readout with VICReg applied upstream of the
  gate instead of at the readout). Such an ablation is *not* on the
  iter_033 path; it is preserved for the constraint-relaxation phase.
- The iter_031 CLTS Part B directional signal (0.59 probe vs 0.44
  random) is what iter_033 will calibrate against measured controls.
  It is a *candidate* behavioral signal, not a result.