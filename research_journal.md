# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — M2 DEFINITIVELY REFUTED across all
    slowness formulations. Transition phase: pivot from slowness-prior to
    identity-discrimination signal.
*   **Active Direction:** With iter_024's double null, the entire slowness-based pathway
    (single-step SFA, weight-swept SFA, multi-step SFA, temporal contrastive NT-Xent) has
    failed to produce identity encoding in z_dyn (delta_R2_color ≥ 0.10 threshold never
    cleared). The next iteration (025) pivots to object-tracking-ID contrastive learning,
    using the physics engine's slot IDs as positive-pair anchors. This is a deliberate
    departure from M2 as written, justified by the empirical refutation.
*   **Next Priority:** Treat iter_025 as a **diagnostic probe to localize the bottleneck**:
    is the failure of slowness-based objectives an objective-level problem (in which case
    ID-contrastive should succeed), or an architecture-level problem (z_dyn cannot encode
    identity through the shared CNN given the soft-argmax centroid head, in which case
    ID-contrastive will also fail)? Both outcomes are informative.
*   **Confidence Score:** 60% (Reduced from 70%. M2's core mechanism is now refuted across
    four objective variants in three consecutive iterations. The decoder-free + identity
    requirement now lacks a validated mechanism in this architecture. The path forward
    is diagnostic, not constructive.)

## 2. Strategic Insights & Lessons Learned
*   **MULTI-STEP SFA AMPLIFIES, DOES NOT RESOLVE, THE SFA-VICReg CONFLICT (iter_024):**
    Multi-step SFA at sfa_weight=10.0 produced 100% collapse, worse than single-step SFA at
    the same weight (2/5 collapse in iter_023). Longer temporal windows give SFA a larger
    gradient that more directly opposes VICReg's variance floor. The standard SFA-literature
    remedy of multi-timescale slowness does not transfer to this architecture.
*   **TEMPORAL CONTRASTIVE (NT-Xent) FIGHTS VICReg WITHOUT WINNING SEMANTICS (iter_024):**
    NT-Xent + VICReg produced 4/5 surviving seeds but the survivors are high-variance noise
    (within_traj_var=0.630), no semantic structure on the color probe. The contrastive
    "push apart different timesteps" signal is incompatible with VICReg's
    variance-decorrelation objective in a non-obvious way.
*   **INVARIANCE-vs-DISCRIMINATION DIAGNOSTIC IS THE CORRECT TOOL (iter_024, METHOD WIN):**
    The shuffled-frame control plus within-trajectory vs. between-trajectory variance
    decomposition cleanly distinguished "smoother that reduces all variance" from "extractor
    that preserves identity-discriminative variance." This diagnostic should be standard for
    any future self-supervised objective candidate.
*   **M2 REFUTATION IS NOW DEFINITIVE (iter_022–024 cumulative):** Four objective variants
    tested, none clear delta_R2_color ≥ 0.10:
      - Single-step SFA, weight sweep (iter_022–023): max 0.064
      - Multi-step SFA, k ∈ {20, 50, 100} (iter_024): max 0.034
      - Temporal contrastive NT-Xent (iter_024): no semantic structure
    The slowness prior is empirically insufficient on RGB+CNN+soft-argmax inputs.
*   **DECODER-FREE + IDENTITY-IN-z_dyn IS NOW UNDERDETERMINED (CRITICAL):** The original
    M2 framing rested on a validated transfer from sml. The transfer has now been refuted
    at the RGB layer. The conjunction (decoder-free × identity-encoding × dual-stream
    shared CNN × soft-argmax centroid head) lacks any validated mechanism. iter_025 must
    decide whether to relax decoder-free, relax shared CNN, or accept identity encoding
    as an unsolved sub-problem.
*   **sml TRANSFER WAS PARTIAL (NOW CONFIRMED EMPIRICALLY):** M1 (pooled VICReg) transferred
    cleanly. M2 (SFA-primary) did not. The sml binary-input result was indeed task-specific,
    vindicating the "measure-before-impose" caution but at the cost of the M2 mandate.
    Prior insights (gradient propagation, ramp strategy, d_max=16 best for color)
    preserved from earlier journal entries.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, FUNDAMENTAL):** No objective tested across
    iter_021–024 produces delta_R2_color ≥ 0.10. This is now the dominant bottleneck;
    everything downstream (gating, motor, generalization) is gated on resolving it. The
    iter_025 ID-contrastive probe is designed to attribute the bottleneck to either the
    objective or the architecture.
*   **Slowness-Prior Loop (CLOSED):** Three iterations of "make SFA work harder" produced
    monotonically improving SFA-mechanism evidence but no improvement on the downstream
    metric. The loop is empirically closed.
*   **Objective-vs-Architecture Disambiguation (NEW):** The next loop to avoid is repeated
    objective swapping without ruling out the architecture. iter_025 must include an
    architecture-independent probe (e.g., supervised linear probe on z_dyn under ID-contrastive
    training) to put a ceiling on what the architecture can encode.
*   **Metric Artifact Loop (CLOSED, iter_023):** C5 abandoned, delta_R2_color is the
    primary criterion.
*   **Gating Design Loop (STALE):** M3 still sidesteps. Not the active concern.
*   **Logistics:** Executor token limits remain a recurring issue across iter_020–024. Not
    blocking but should be tracked.

## 4. Alternate Research Paths
*   **Object-Tracking-ID Contrastive (IMMEDIATE, iter_025):** Use physics-engine slot IDs to
    build positive pairs (same object, different time) and negative pairs (different
    objects, same time). This is a stronger supervisory signal than self-supervised
    contrastive but maintains the "no pixel decoder" property. It is also a **diagnostic
    probe**: if z_dyn cannot encode identity even under this strong signal, the bottleneck
    is the architecture, not the objective.
*   **Supervised Linear Probe on z_dyn (DIAGNOSTIC, parallel to iter_025):** Train an
    ID-contrastive z_dyn, then fit a linear probe predicting object color/size from
    z_dyn alone. The probe accuracy is the architecture's ceiling — if it is low even
    under direct supervision, z_dyn cannot encode identity through this CNN.
*   **Separate Identity Encoder (HIGH PRIORITY if ID-contrastive on shared CNN fails):**
    Decouple the encoder for z_coord (current CNN + soft-argmax) from a second encoder for
    z_dyn. This relaxes the shared-CNN constraint that may be the structural bottleneck.
    Expensive but clean.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR style) (MEDIUM PRIORITY):** Only
    revisit if ID-contrastive succeeds, to recover self-supervision after using ID labels
    as a diagnostic.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT):** A small pixel decoder
    restricted to z_dyn (not z_coord) would directly force identity encoding. This breaks
    the decoder-free principle and should only be considered if all decoder-free paths
    fail.
*   **Micro-Columns (DEFERRED per semantic caution):** Still gated on a working objective.
    The objective is the active gap.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.