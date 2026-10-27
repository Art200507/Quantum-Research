# Existing-project audit — 25 October 2026

## Evidence and technically working behavior

Inspected all seven Python modules, requirements, README, and saved JSON. Neither
workspace nor project is a Git repository; no applicable AGENTS.md was found in
workspace/project or checked ancestor locations. Original edited sources are
copied under `originals/`; historical results are untouched. No account files
were opened and no service/account initialization or hardware submission occurred.
Installed versions match all four pins; full scientific dependency versions are
in `results/metadata.json`. No additional installation was needed.

`run.py` constructs labeled submissions, compiles against FakeTorino locally,
extracts `data.meas` counts, analyzes and saves. The runtime executor Sampler
import exists in installed 0.51.0; a small local run verifies its result schema.
This is not verification of cloud execution or job usage APIs. Real mode retains
both `--real` and a literal `yes` prompt before submission.

The historical `results/20261023T194020Z.json` explicitly says `dry_run`,
`fake_torino`, 1000 shots, null hardware job/usage. Bell support fraction .977;
GHZ support .966; 5-qubit mirror return .772. These are simulated measurements,
not physical-device evidence. Its 20-qubit ideal return is 1; 50/100 have no
execution data. The saved Python-PRNG bit-0 p=.001565 is a single unseeded
frequency fluctuation, not evidence quantum randomness is superior. Its eight-bit
Bonferroni threshold at family alpha .01 is .00125, so it does not cross it.

## Scientifically supported conclusions

| Experiment | Actual measured quantity | Supported interpretation | Unsupported inference |
|---|---|---|---|
| Hadamards and measure | Marginal bit proportions, frequency chi-square, coarse byte histogram | Agreement/disagreement with equal bit frequencies under assumed sampling model | Quantum origin, entropy certification, independence, cryptographic unpredictability or advantage over Python |
| Bell/GHZ | Z-basis populations and mass on expected support | Population leakage/correlation sanity check | Measured entanglement, coherent superposition or state fidelity |
| U then inverse | All-zero return fraction and mean Hamming weight | End-to-end diagnostic including preparation, gates and readout | Process fidelity, component error rates, universal fidelity, classical intractability |

For a perfect computational-basis measurement, mirror P(0...0) equals overlap
`<0|rho|0>` (squared state fidelity convention for this pure target). Here noisy
readout is included, so the recorded outcome fraction need not equal the
premeasurement state's overlap. It certainly is not the fidelity of U or its
intermediate state. The variables historically named `fidelities` are just
return probabilities. Bell `backend_expected_fraction` is support probability,
not a fidelity estimate. The ideal references are sampled counts, not exact
state reconstruction.

The separable mixture (|00><00|+|11><11|)/2 has the same Bell Z histogram as
|Phi+>. Add XX and YY settings: F_Phi+=(1+<XX>-<YY>+<ZZ>)/4; a confidence-aware
lower bound F>1/2 certifies entanglement with trusted basis rotations and
measurement. For GHZ3, measure Z populations plus XXX, XYY, YXY, YYX:
F_GHZ=(P000+P111)/2+(<XXX>-<XYY>-<YXY>-<YYX>)/8. A lower bound >1/2 witnesses
genuine tripartite entanglement. Account for shot allocation, multiple settings,
and readout/basis errors; ideal simulator preparation alone certifies nothing
about an unknown hardware state. These formulas follow directly by expanding
the corresponding pure-state projectors in Pauli operators.

## Cancellation and classical simulation

`checks/audit_local.py` compiled the same 8-layer mirrors, seed 1234+N and compiler
seed 17. Actual results in `results/compiled_mirrors.json`:

| N | optimization | CZ without midpoint barrier | CZ with barrier | with-barrier total depth |
|---|---|---|---|---|
| 5 | 3 | 0 | 32 | 59 |
| 20 | 3 | 0 | 152 | 76 |
| 50 | 1 | 312 | 392 | 81 |
| 100 | 1 | 644 | 792 | 84 |

A 5-qubit all-to-all U/CX compilation gives depth 0 without barrier versus 34
with it; exact return is 1 within numerical precision for both. Thus the barrier
retains nontrivial entangling work for these settings. It does not preserve each
original gate. The 50/100 previews use a different optimization level, so do not
compare their depths with level-3 hardware circuits as equivalent workloads.
Compiler variability remains uncharacterized; original production helper has no
seed_transpiler. Full physical-layout equivalence of all large ISA circuits was
not checked by statevector simulation.

Generic rotation angles make the ensemble non-Clifford; pure stabilizer simulation
is not generally exact. Clifford special cases remain easy. Eight layers on a
nearest-neighbor chain have bounded entanglement across each cut independent of
chain length, so matrix-product/tensor-network approaches matter. Local observables
also have small causal cones. Full noisy all-zero probability differs from a local
observable, but can be contracted using finite-depth circuit structure; cost can
be exponential in depth without being exponential in qubit count. No such large
noisy contraction was benchmarked here. An ideal return is known algebraically
at any N. Full-state memory arithmetic is not a lower bound on predicting an
observable or sampling structured circuits. No classical difficulty claim survives.

## Noise, statistics, plots and unverified assumptions

FakeTorino is a bundled calibration snapshot and associated Aer model, useful for
rehearsal rather than a validated description of today's device. Calibration drift,
coherent/correlated noise, crosstalk and scheduling effects can be absent or
misrepresented. Noise need not require one trajectory per shot: Aer supports
density matrices and other methods. Cost depends on method, noise structure and
measurement placement. Local caps are resource policy, not physical thresholds.

The chi-square survival function is evaluated exactly for its asymptotic reference
distribution; the finite-binomial test is not exact. Per-bit IID assumptions matter;
flattening correlated qubits violates the pooled test's independence assumption.
Counts destroy shot order; reconstructed byte values must not be used for temporal
randomness tests. QRNG positions are printed-string positions, not explicitly
logical qubit labels (Qiskit strings are high index first). A 32-bin histogram
is descriptive, not a 256-outcome uniformity test. At 1000 shots a full byte test
has expected counts only 3.906 per category, requiring care with asymptotics.

Original plots have no uncertainty intervals or repeats and overwrite filenames.
At 1000 IID shots a proportion's worst-case standard error is .0158; at 128 it is
.0442. Hamming weight has within-shot correlations and needs its own sample
variance. Wilson intervals or exact binomial intervals and repeated circuit/layout
seeds are needed for inferential benchmarking. No shot count by itself guarantees
power. One random mirror at each size confounds circuit identity, size and mapping.
A good inverse return may hide coherent error cancellation; it is no universal
predictor of a target experiment.

## Corrections made and remaining work

Removed unsupported beyond-classical claims and fidelity plot labels, corrected
QRNG flagging for eight simultaneous marginal tests, distinguished asymptotic
p-values in documentation, withdrew unvalidated 30–90 second QPU estimates, and
removed “instant” runtime claims. Fixed `--out` so JSON and plots go together and
created requested output directories; reject nonpositive shots/qubit sizes and
unknown experiment names. Hardware confirmation remains intact. `setup_account.py`
hides the key, but displays the CRN input; the handoff's claim both were hidden
was incorrect.

The working demo API and old JSON field/plot filename compatibility remain. Internal
legacy variable names/filename `mirror_fidelity_vs_n.png` are historical, while
new plot text states the actual metric. Existing saved plots are historical and
were not relabeled. The demo remains educational: no uncertainty retrofit, temporal
RNG testing, tomography, or hardware verification was silently added. Research
additions provide explicit error accounting separately.

IBM's current [official plan page](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview)
confirms up to 10 minutes per rolling 28 days and describes an opt-in additional
allocation for eligible active users. Eligibility and remaining allowance were not
checked. Batching changes job organization; it does not establish low usage.
