# RDF Milestone Review — Iteration 027 — Null Result: Shared CNN Backbone Is Not The Primary Cause Of z_dyn Collapse

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (carried forward from iter_026 journal entry
and the iter_027 plan):
  "The shared CNN backbone is the primary cause of z_dyn collapse under
   JEPA+VICReg. A separate dedicated backbone for z_dyn, with all other
   hyperparameters held constant, will reduce the collapse rate to ≤10%
   over ≥10 seeds (Arm B)."

Pre-declared falsification criterion:
  "If Arm B's collapse rate is ≥20% under the dual collapse criterion
   (eval-std < 0.5 OR train-std < 0.5), the shared-backbone hypothesis
   is rejected."

## 2. Experimental Protocol
- Encoder: NonParametricJEPASpatial, d_max=8, d_t=3.
- Arm B (the hypothesis vehicle): separate backbone for z_coord and
  z_dyn (135,608 parameters), full JEPA+VICReg objective on both
  streams, sim_weight=25, var_weight=25, cov_weight=1.
- Buffer capacity: 4000 (held constant from iter_026 to control the
  flagged buffer-size confound).
- Optimizer: lr=3e-4, batch_size=32, 8000 training steps.
- Matching: Hungarian-primary.
- Seeds: n=10.
- Control arms run in the same iteration: shared-backbone baseline
  (reference: 30–40% from iter_026); Arm C (separate backbone,
  `mask_dyn_sim=True`).
- Held constant between Arm B and Arm C: parameter count (135,608),
  backbone topology, hyperparameters, seed bank, matching procedure.

## 3. Observed Quantities
- Arm B (separate backbone, full JEPA+VICReg): **30% collapse rate**
  over 10 seeds under the dual criterion (eval-std < 0.5 OR
  train-std < 0.5).
- Shared-backbone reference (iter_026 A0/A1): 30–40% collapse.
- Falsification threshold: ≥20%.
- Difference between Arm B and the shared-backbone reference: within
  seed-noise (n=10 per arm).

## 4. Verdict
**REFUTED.** Arm B's 30% collapse rate clears the pre-declared
rejection threshold (≥20%). The shared CNN backbone is not the
primary cause of z_dyn collapse under the current JEPA+VICReg
objective. Architectural decoupling at the encoder level alone does
not stabilize z_dyn.

## 5. Construction-vs-Empirical Note
The null on Arm B is genuinely empirical: the falsified prediction
was about gradient-pathway competition between z_coord and z_dyn in
shared parameters. Removing the shared parameters and observing no
improvement is real information about the mechanism — it tells us
the competition is not occurring at the shared-encoder gradient
level, or that it occurs but does not dominate the collapse
dynamics.

Separately, this iteration's Arm C (`mask_dyn_sim=True`, 0% collapse)
is **not** promoted to a finding in this report. The Arm C result has
a partial construction-versus-empirical concern: VICReg's variance
hinge directly enforces the same std quantity the collapse criterion
measures, so a VICReg-only z_dyn maintaining std ≥ 1 is partly what
the loss function is being told to do. Arm C is recorded as a
suggestive within-architecture ablation in the journal, requiring
iter_028 confirmation (shared-backbone version, replication on a
different seed bank, ±10% robustness check).

## 6. Limitations
- This result does **not** show that the shared backbone is irrelevant
  — only that it is not the dominant cause. Subtle effects (e.g.
  interaction with optimizer momentum, or with larger d_t) may still
  exist.
- n=10 per arm; differences smaller than ~14 percentage points are
  within seed noise.
- This result does **not** establish what *is* the primary cause.
  The cross-iteration synthesis (iter_026 + iter_027) suggests
  `sim_loss_dyn`–VICReg competition is a leading candidate, but
  confirmation requires iter_028's missing control arm
  (`mask_dyn_sim=True` on shared backbone).
- The collapse criterion itself (eval-std < 0.5) is conservatively
  chosen but is one of multiple possible thresholds; the
  train-vs-eval std discrepancy flagged in iter_026 remains an open
  architectural signal.
- The buffer-size confound (iter_026 → iter_027 buffer=4000) is
  controlled within this iteration but limits comparability with
  earlier iter_025-and-prior results.