Update `src/pre_registration.md` to add a v2 revision section at the end. APPEND the following text to the existing file (do not overwrite what's there):

```

---

## v2 Revision (post v1 falsification)

v1 MAPE was falsified: PASSIVE=0.597 < RANDOM=0.999 < ORACLE=1.005 (inverted ordering).
Root cause: pointer-object mass estimates (m_i = 10*(-Δv_pointer)/Δv_obj) are extremely
noise-sensitive when Δv_obj is small, and hundreds of such rows overwhelm the
least-squares system. Object-object collision ratios are more stable.

v2 replaces MAPE with MALRE (Mean Absolute Log-Ratio Error) computed from the MEDIAN
of mass-ratio estimates from object-object collisions only. No velocity noise injection.

### v2 Results (8 seeds × 3 conditions = 24 runs)

MALRE (lower is better):
- ORACLE: 0.503 ± 0.151
- RANDOM: 0.534 ± 0.345
- PASSIVE: 1.333 ± 0.000

### v2 Gates (all PASS)
- G1 (PASSIVE-ORACLE ≥ 0.3, CI ≥ 0.1): gap=0.830, CI=[0.727, 0.939] — PASS
- G2 (PASSIVE-RANDOM ≥ 0.1, CI > 0): gap=0.799, CI=[0.530, 1.004] — PASS
- G3 (ORACLE < RANDOM < PASSIVE): 0.503 < 0.534 < 1.333 — PASS
- G4 (ORACLE-PASSIVE coverage ≥ 0.2): gap=0.50 — PASS

All sanity checks S1-S5: PASS

### v2 Caveat (critical for iter_035)
The ORACLE vs RANDOM gap is only 0.031 (not significant). ORACLE beats RANDOM on
only 3/8 seeds. The PASSIVE gap is driven entirely by coverage (PASSIVE has no data
for most object-object pairs, getting max penalty), NOT by estimation quality.
The metric discriminates active-vs-passive but NOT targeted-vs-random within the
active regime. For iter_035, this means:

1. The MALRE metric CAN detect whether an agent explores at all (active vs passive)
2. The MALRE metric CANNOT discriminate between good and bad targeting within active exploration
3. iter_035 needs EITHER a different metric for the ORACLE-vs-RANDOM comparison, OR must
   accept that the benchmark's discriminating power is limited to the active/passive boundary

The benchmark is validated as a COVERAGE DISCRIMINATION test (responds to whether the
agent explores), not as a TARGETING QUALITY test (does not respond to how well the
agent targets under-explored regions). The pre-registered hypothesis is confirmed in
the limited sense that the metric discriminates targeted-from-untargeted coverage of
the collision manifold, which is a necessary precondition for measuring perception —
but the active regime (ORACLE vs RANDOM) is underpowered.

### v2 Code
- `src/run_iter034_v2.py` — MALRE benchmark implementation
- `archive/iter_034/results_v2/` — results
```

Read the file first, then append this text to the end. Do NOT overwrite existing content.