# Pre-Registration: Iteration 030

## ARM 1 — Integration Smoke-Test

### Hypothesis
A representation with ΔR²_color ≈ 0.27 (SFA+VICReg on separate backbone, 0% collapse, iter_029 Arm B) produces CLTS motor behavior that is functionally adequate for Pillar D (thalamic gating) and Pillar E (motor probing). The weaker representation (ΔR²_color ≈ 0.04, VICReg-only, iter_029 Arm A) may also be adequate, which would indicate identity decodability does not bottleneck downstream behavior and M2 should be demoted.

### Falsification Criteria
Three pre-registered gates, evaluated per-seed with paired comparisons:

**G1 (Tracking Functionality):** CLTS-SFA mean |pointer_pos - attended_centroid| < 20 pixels over a 2000-step closed-loop evaluation (steps 200-2000, after EMA warmup). CLTS-VICReg must also meet this threshold. This gate tests whether z_coord centroid readout supports functional PD tracking.

**G2 (Attention Validity — Collision):** Within 15 steps after a detected collision (|delta_v| > 2.0 for any object pair), CLTS attention switches to a collision-involved channel. We measure per-seed collision-switching-rate = fraction of collision events where CLTS attention is on a collision-involved channel within 15 steps. CLTS-SFA must have collision-switching-rate ≥ CLTS-Random + 15pp AND ≥ CLTS-Frozen + 15pp. The same applies to CLTS-VICReg. 

With n_seeds=10, estimating ~5-10 collision events per 2000-step run, each seed contributes a binomial outcome. Under null (no advantage), per-seed rate difference has expected 0. The 15pp threshold is a meaningful behavioral improvement. With 10 seeds and paired comparisons, a consistent 15pp advantage across seeds rejects the null at p < 0.05 via sign test (≥9/10 seeds favoring CLTS under null: p = 0.010).

**G3 (Causal Sensitivity — Mass Perturbation):** At step 1000, object-0 mass is tripled and the pointer is forced near object 0 (pointer_pos = obj_0_pos ± 5) with a push. Within 20 steps, CLTS attention must switch to object 0's channel. Per-seed: 1 if switch occurs, 0 if not. CLTS-SFA switch-rate must ≥ CLTS-Random switch-rate + 15pp AND ≥ CLTS-Frozen switch-rate + 15pp. With 10 seeds and 1 perturbation per seed, under null (1/3 chance per random), expected random switch-rate is ~33%. CLTS must achieve ≥48%.

### Decision Rule
- ≥2 of 3 gates pass for CLTS-SFA → representation sufficient → project advances to Phase 2/3
- 0-1 gates pass for CLTS-SFA → representation insufficient → objective hunt justified (ARM 2 becomes critical)
- CLTS-SFA passes AND CLTS-VICReg passes → identity decodability does NOT bottleneck downstream behavior → M2 demoted
- CLTS-SFA passes AND CLTS-VICReg fails → identity encoding DOES matter → 0.30 search justified
- Both fail → representation truly insufficient
- CLTS-VICReg passes but CLTS-SFA doesn't → investigate anomaly

### Controls (mandatory)
1. **CLTS-Random**: Same CLTS controller but token_locus selected uniformly at random from d_t channels (replacing argmax of normalized surprise). PD tracking and push logic identical.
2. **CLTS-Frozen**: Token_locus held at channel 0 throughout evaluation. Isolates contribution of attention switching from PD tracking.

### Seeds
10 fresh seeds (101, 103, 107, 109, 131, 137, 139, 151, 157, 163) from iter_029 union bank. Hard seeds 53, 71 reported separately, NOT included in gate calculations.

---

## ARM 2 — M2 Decisive Test

### Hypothesis
Temporal identity contrastive binding (D1) or variance-ramped SFA (D2) achieves mean ΔR²_color ≥ 0.30 on a 30-seed union bank with variance-stability (lower bound of the 95% CI ≥ 0.18, anchored to iter_027 Arm C result).

### Falsification Criteria
- **D1 falsified**: mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on the 30-seed bank
- **D2 falsified**: mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on the 30-seed bank
- **Both D1 and D2 falsified**: ΔR²_color ≥ 0.30 is unachievable by any decoder-free objective tested; project accepts current representation quality per ARM 1 verdict

### Seed Bank
30 seeds: original 10 (7, 17, 31, 53, 71, 83, 97, 113, 127, 149) + fresh 10 (101, 103, 107, 109, 131, 137, 139, 151, 157, 163) + new 10 (173, 179, 181, 191, 193, 197, 199, 211, 223, 227). Hard seeds 53, 71 flagged.

### Preservation Rules
- Separate backbone + collapse-avoiding config (mask_dyn_sim=True, coord_vicreg=True)
- d_t=3 frozen, GDASR log-only (M3)
- Pooled batch VICReg (M1)
- No positional encoding (cross-objective regularity from iter_013 and iter_021)
- Hard seeds 53/71 reported separately, not averaged away
- Report σ alongside mean for all metrics (≥5 seeds minimum per condition)