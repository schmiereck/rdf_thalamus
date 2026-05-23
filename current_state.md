# Current Research State

## Goal
The primary goal is to design, simulate, and evaluate a novel neural architecture ("Thalamus") that achieves hierarchical abstraction without generative decoders inside a 1D physics environment. Specifically, the network must:
- Dynamically allocate new representational dimensions for unexplained inputs.
- Learn temporal prediction in latent space (no backprop-through-time over pixel reconstruction).
- Form a thalamic gating mechanism (plasticity and attention routing) focusing on the epistemically most relevant entities.

Learning is driven by local temporal prediction error ("surprise"), with anti-collapse (e.g., VICReg) and anti-dark-room mechanisms (using environmental variation and active exploration).

## Confirmed
- Workspace environment successfully initialized (iter_001.2).
- CPU-only execution environment validated: AMD Ryzen 5 7535U, 16.0 GB RAM, PyTorch 2.12.0+cpu, NumPy 2.4.6.

## Refuted
- None (baseline phase).

## Best Result
- None (baseline phase).

## In Progress
- Setup and architectural design of the Phase 1 representation base.

## Open Questions
- What dynamic dimension recruitment mechanism (Online PCA, Hebbian recruitment, SOINN, GNG, or MoE) is most stable under changing local gradients?
- What anti-collapse constraint (VICReg-style variance/covariance vs. BYOL-style stop-gradients vs. NGC error neurons) provides the strongest defense against trivial solutions in this 1D setup?
- What temporal modeling mechanism (delay-line context vs. local GRU) is most sample-efficient for 1D physics tracking?
- How do we design the cross-scale energy normalization to prevent higher-level surprise from being systematically masked by lower-level pixel-level surprise?
