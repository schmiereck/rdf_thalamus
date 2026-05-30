# Research Manager Log - Iteration 031

## Iteration 031 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for the Planner, iter_031)

1. Retire ΔR²_color ≥ 0.30 as the gating metric. Across iter_023–030 every decoder-free objective in the slowness/VICReg/contrastive family has failed it, including the full M2 configuration with confounds controlled. Continuing to sweep objectives against this proxy is now an exploratory loop with diminishing information yield. The next iteration must either (a) re-derive a *downstream-functional* gate that the project actually needs identity encoding for, or (b) explicitly justify why ΔR²_color remains the right target after eleven iterations of refutation. Do not propose a fourth slowness variant against the old gate.

2. Treat M2 as empirically falsified for this task and pre-register the mandate-revision decision. The cross-architecture convergent null (iter_023–024 shared backbone, iter_029 separate backbone, iter_030 variance-ramped D2) is now sufficient to overturn the transferred sml mandate in the Thalamus domain — the "scope of transfer" caveat in Section 1.1 has been triggered. The plan must state, in advance and in writing, what the replacement primary representation objective is (candidates: BYOL-style identity targets, object-tracking contrastive with object-level — not batch-level — matching, or a controlled relaxation of the decoder-free constraint with VICReg+reconstruction as the iter_030 analysis itself flagged). Pick one as primary with a falsification criterion; do not run three more arms hoping one survives.

3. Rebuild the protocol *before* declaring any positive functional claim. The iter_030 ARM 1 ceiling effects (G2/G3 switch rates ≈1.0 for random) and the unrealistic 20-pixel tracking gate mean the project currently has no calibrated downstream measurement. Any new representation objective must be evaluated against a protocol whose *random-policy baseline is empirically measured first* and whose gates are set to exceed that measured baseline by a pre-declared margin — not against thresholds picked in advance of knowing what chance looks like. Treat the protocol-calibration run as a required preamble, not an aside.

---

## Iteration 031 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30
across 20 seeds (lower 95% CI ≥ 0.18), establishing that the architecture supports identity
encoding and that the M2 failure was objective-specific (all decoder-free objectives
insufficient for this architecture's mean-readout z_dyn stream), not architectural.
Additionally, reconstruction-trained models produce centroid MSE < 120 (improved over the
~160 baseline from VICReg-only and SFA+VICReg arms).

Specifically: a deconv decoder head on the dyn backbone's spatial features (B, d_max, 8)
→ (B, 3, 128) with loss = recon_weight × MSE(x_recon, x_input) + var_weight × VICReg(z_dyn)
+ cov_weight × VICReg(z_dyn) + (coord_vicreg=True) VICReg(z_coord) shapes z_dyn to carry
object identity information, as measured by the same ΔR²_color linear probe used across
iter_020-030.

**Proposed Falsification Criterion:**
If Reconstruction+VICReg achieves mean ΔR²_color ≤ 0.275 (the best decoder-free result from
iter_029 Arm B, SFA+VICReg sfa_weight=5.0, 20 seeds), then the mean-readout z_dyn architecture
itself constrains identity encoding regardless of objective class, and the project must
redesign the z_dyn readout mechanism (e.g., centroid-gated readout from iter_027 Arm A') or
encoder architecture before any further objective work. This would be a fundamental architectural
finding, not an objective finding.

**Proposed Method:**
## Part A: Reconstruction+VICReg Ceiling Probe (PRIMARY)

### A1: Model Implementation
Create `src/models_recon.py` containing `ReconVICRegSeparateDyn`:
- Encoder: `SeparateDynEncoder` (existing separate coord + dyn backbones, unchanged)
- Decoder: Deconv head on dyn spatial features a_dyn (B, d_max, 8) → (B, 3, 128)
  Architecture: ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) → ReLU → 
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Loss: recon_weight × MSE(x_recon, x_target) + var_weight × [VICReg_var(z_dyn) + VICReg_var(z_coord)]
  + cov_weight × [VICReg_cov(z_dyn) + VICReg_cov(z_coord)] + sim_weight × predictor_loss
- Predictor: DualStreamPredictor with stop-gradient on encoder output (surprise readout only)
- All attributes needed for evaluation pipeline: encoder, d_t, d_max, sub_features,
  color_probe_weight, color_probe_bias, id_contrastive_proj, gdasr_growth_points

### A2: Quick Hyperparameter Scan
- recon_weight ∈ {10.0, 25.0, 50.0}
- 3 seeds per weight (7, 31, 97), 2000 steps each = 9 quick runs
- var_weight=25.0, cov_weight=25.0, sim_weight=1.0, coord_vicreg=True
- Select best recon_weight by ΔR²_color for full run

### A3: Full Training (20 seeds, union bank)
- Seeds: 10 original [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] + 10 fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
- 8000 steps, batch_size=32, lr=3e-4, d_t=3, d_max=8
- pos_encoding="none", coord_vicreg=True
- GDASR in log-only mode (no recruitment)

### A4: Evaluation (identical pipeline to iter_029/030)
- ΔR²_color (primary): linear probe from z_dyn to object color, channel-object matched
- Centroid MSE: soft-argmax centroid decoding
- Collapse rate: per-dim std < 0.5 threshold
- VICReg health: per-dim std, mean absolute cross-correlation
- Reconstruction MSE
- All metrics reported with 95% CI across 20 seeds

### Comparison Baselines (from iter_029, no re-run)
- VICReg-only: ΔR²_color = 0.045 (20 seeds)
- SFA+VICReg sfa=5.0: ΔR²_color = 0.275 (20 seeds)

## Part B: Protocol Calibration (REQUIRED PREAMBLE)

### B1: N=2 Collision-Sparse CLTS Evaluation
- Using existing VICReg-only checkpoint from iter_029 (best tracking: 36.22 px)
- Environment: PhysicsSandbox(N=2), fewer collisions than N=3
- 3 conditions per seed (5 seeds): surprise-driven, frozen (locus=0), random
- 2000 evaluation steps per condition
- Measure FIRST: random/frozen baseline for tracking error, collision count/100 steps,
  collision attention selectivity (fraction of post-collision steps where colliding channel is attended)
- Report measured random baselines explicitly
- Define data-driven gate formulation: active condition must exceed random baseline by
  a pre-declared margin (e.g., tracking error ≤ random − 1σ, or collision selectivity ≥ random × 1.5)

### B2: Subtle Mass Perturbation Test
- In same N=2 environment, at step 1000: mass of object 0 changes by 1.5× (not 10×)
- Measure: perturbation attention selectivity (fraction of steps post-perturbation where
  changed object's channel is attended)
- Compare surprise-driven vs random baselines

## Mandate Revision (PRE-REGISTERED)
- If ceiling probe SUCCEEDS (ΔR²_color ≥ 0.30): M2 revised from "SFA+VICReg as primary"
  to "Reconstruction+VICReg as primary representation objective, decoder-free constraint
  relaxed as pragmatic compromise. SFA demoted to comparison baseline B1. Surprise readout
  retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free
  alternatives approaching the reconstruction ceiling."
- If ceiling probe FAILS (ΔR²_color ≤ 0.275): "Mean-readout z_dyn architecture is
  insufficient for identity encoding under any tested objective class. M2 revision pending
  architectural redesign. Priority: centroid-gated z_dyn readout (iter_027 Arm A' showed
  directional improvement) or increased d_max."

---

## Iteration 031 -> Planner [Strategic Guidance]

Manager's Note (iter_031) — Critique of Proposed Plan

**Endorsed with two structural corrections and one scope reduction.** The pivot to Reconstruction+VICReg is well-motivated: per the iter_030 analysis, it is the one untested objective class that sml itself rated highest (83% vs SFA's 82%), and the hypothesis "the failure is objective-specific, not architectural" is exactly the disambiguation needed to either rescue Phase 0 or condemn the readout. The two-part structure (ceiling probe + protocol calibration) is the right shape. However:

---

**1. The hypothesis as stated is partially circular and will smuggle in a constructional result — rewrite it.** A deconv decoder trained with reconstruction MSE is *being told to preserve information* via a gradient that flows back through `z_dyn` from a pixel-MSE target. ΔR²_color of the resulting `z_dyn` is then a downstream readout of "did the bottleneck preserve color." If the bottleneck is wide enough (`d_max=8` channels × spatial dim 8 = 64 scalars representing 3×128=384 pixel values, ~6× compression), reconstruction *must* preserve color to minimize MSE, and the linear probe *must* pick it up. That isn't an empirical finding about whether the architecture can encode identity — it's a verification that 64 scalars can hold the color of three objects, which is true by counting. **Reframe the hypothesis as a two-sided question**: (a) does ΔR²_color clear 0.30 *and* (b) does it do so with a non-trivial margin over a strawman that is forced to preserve information (e.g., random-encoder + frozen-decoder, or a tiny-`d_max` ablation that *should* fail). Without (b), a "pass" tells you only that the construction works, not that the *training* did anything. Add at minimum a `d_max=2` (under-capacity) and a `d_max=8` random-encoder control to the 20-seed run so the result has a contrast to interpret.

**2. The falsification criterion is mis-set.** As written, the bar is "≤ 0.275" (the iter_029 SFA mean), but iter_029 had σ = 0.577 — the SFA result itself is not stable. Reconstruction beating an unstable mean is not interesting. Restate F1 as: **mean ΔR²_color ≥ 0.30 AND lower 95% CI ≥ 0.18** (the iter_027 Arm C anchor, the most stable empirical floor we have), matching the variance-stability discipline the Journal already adopted in iter_030. Pre-declare a *failure case* too: if the lower CI is below 0.18 even with mean above 0.30, the result is "directionally positive, not robust" — same verdict as iter_029 SFA — and triggers a variance investigation, not a mandate revision. Without this, an unstable pass becomes another contested mandate.

**3. Drop the 9-run hyperparameter scan (A2) or make it pre-registered.** Picking `recon_weight` by ΔR²_color on 3 seeds and then evaluating the winner on 20 seeds is a **Gate-3 (Parameter-Tuning Hygiene) violation by construction** — exactly the failure the goal document warns about: "if a desired effect only appears after lowering thresholds, increasing coupling, or expanding ranges *during* a sweep, treat that as a negative result." Either (a) pick `recon_weight` from a *non-target* metric (reconstruction MSE alone, or VICReg health) and accept whatever ΔR²_color falls out, or (b) drop A2 entirely and run `recon_weight=25.0` (the natural midpoint) on the full 20-seed bank. Option (b) is cleaner.

**4. Pre-registration mandate (procedural).** Before any code runs, write the rewritten hypothesis, the corrected F1/F2/F3 gates, the controls from point 1, and the locked `recon_weight` from point 3 into `src/pre_registration.md`. The Orchestrator will commit this file before execution. Sub-agents must read it and adhere to it — no in-flight gate adjustment. Particularly important here: pre-commit the language for *both* outcomes, so a constructional pass is reported as "the architecture has sufficient bottleneck capacity to preserve color under a supervised pixel target" rather than "the architecture supports identity encoding."

**5. Language discipline for the report.** Replace "establishing that the architecture supports identity encoding" with "is consistent with the architecture having sufficient capacity to preserve color information under a supervised target." Reconstruction-trained representations carrying color is the expected outcome of the loss design, not an empirical discovery about the architecture's representational *learning*. The genuinely empirical question is the *margin* over the random-encoder and under-capacity controls.

**Part B (protocol calibration) is endorsed as written** — measuring random baselines before declaring gates is exactly what was missing in iter_030, and N=2 collision-sparse is the right reduction. One small ask: pre-register the margin formula in `pre_registration.md` *before* running B1, not after seeing the random numbers, to keep the gate honest.

---

