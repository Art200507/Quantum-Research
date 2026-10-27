# Candidate questions, challenges and recommendation

These are proposed evaluations, not established gaps. Prior-work IDs refer to
LITERATURE.md. Ranking is tailored to an intermediate Python programmer at UB
who has not studied linear algebra, probability or quantum mechanics. Weekly
availability is unknown; the schedule assumes 6–8 hours/week provisionally.

| Rank | Question | Current status | Value / evidence of gap / feasibility / reproducibility / speed (1–5) | Hardware |
|---|---|---|---|---|
| 1 | When does fixed-budget ZNE beat a tuned raw product formula? | Focused methodological replication; pilot implemented | 4 / 1 / 5 / 5 / 5 | Unnecessary for pilot, optional later |
| 2 | When do compiled layouts change which circuit is most accurate? | Replication with possible incremental extension | 3 / 2 / 4 / 4 / 4 | Optional; essential for claims about actual device layout quality |
| 3 | Do mirror diagnostics predict held-out dynamics error? | Existing benchmarking problem; possible narrow extension | 4 / 2 / 3 / 4 / 3 | Optional locally, valuable for transfer claims |
| 4 | How fragile is mitigation to noise-scaling mismatch and drift? | Robustness replication; possible incremental extension | 4 / 2 / 3 / 4 / 3 | Optional, necessary for empirical drift claims |
| 5 | Which classical representation is best for the tested observable? | Resource/accuracy replication | 3 / 1 / 3 / 5 / 3 | Unnecessary |

Scores are judgment, not bibliometric measurements. No candidate is presently a
verified novel contribution. Candidate 1 wins on learning and informative results,
not on the evidence for novelty. Candidate 2 is the fallback.

## 1. Fixed-budget mitigation versus choosing a better circuit

**Question/motivation.** At fixed evolution time and total estimation resources,
can two-scale linear ZNE beat the best unmitigated first- or second-order Trotter
circuit for Z0 and Z0Z1 in an open spin chain? Users choose circuits as well as
mitigation; comparing only against the same raw depth can mislead.

**Closest work and challenge.** E19 already studies the Trotter/noise optimum;
C23 studies resource-limited mitigation; S26 directly studies finite-shot MSE;
Z26 jointly treats algorithmic and physical error. Searches for optimal noisy
Trotter steps, fixed-shot ZNE, and joint extrapolation found strong overlap.
Thus neither an optimum nor a bias–variance crossover is novel.

**Possible unresolved part/contribution.** A carefully reproduced joint depth,
shot and observable comparison with a declared continuous-H target. A subsequent
extension could test whether a cheap training protocol selects a good method on
held-out times/noise profiles after charging selection costs. Whether that is
new requires full code/method comparison with S26 and Z26 and supervisor advice.

**Objective/hypothesis.** Find a reproducible setting where same-depth ZNE lowers
bias but increases MSE, and test whether apparent gains survive tuned-raw and
resource-matched baselines. Both outcomes are useful. Predeclare thresholds;
a provisional follow-up criterion is >=20% held-out MSE improvement with a
confidence interval excluding no gain after selection/calibration costs.

**Minimum experiment/baselines.** The completed 3/4-spin grid, exact evolution,
noiseless digital circuit, raw noisy estimates, equal and 3:1 ZNE allocations;
fixed-shot and abstract gate-work comparisons. Follow up with actual folding,
noise mismatch, held-out t values and a paid-for selection stage.

**Knowledge/resources.** Complex vectors, matrix multiplication, Pauli operators,
matrix exponential, expectation/variance and Monte Carlo; local pinned Python
environment. Hardware unnecessary for replication.

**Confounders/failure.** Bias cancellation, oracle p scaling, uncharged calibration,
noise models without coherent errors, use of truth to tune, integrability, and
native compilation that merges gates. A negative result—no gain over tuned raw
under fair costs—is useful and should determine a stopping decision.

**Two weeks / 8–12 weeks.** Understand and reproduce one result and verify the
Pauli estimator formula. Then replace oracle amplification, test held-out
selection, compare prior code, and ask a supervisor whether any residual problem
warrants an extension. Publication is not the default endpoint.

## 2. Compilation and layout versus local-observable error (fallback)

**Question/motivation.** For a fixed logical chain and observable, how well do
compiled two-qubit count and calibration-weighted cost rank observable error
across layouts and compiler seeds? Local measurement may depend on only part of
the circuit.

**Closest work/challenge.** C23 shows device emulation/connectivity effects;
application-oriented benchmarking is extensive. Search results included AppQSim,
which must be fully inspected before narrowing this direction. SABRE/compiler
heuristics themselves would not be a contribution.

**Unresolved possibility/contribution.** A transparent error-rank dataset for
one target family, with train/test separation, might expose observable-specific
misranking. No gap established; hardware calibrations are noisy surrogates.

**Hypothesis/evaluation.** A simple causal-cone weighted gate metric predicts
held-out absolute observable error more accurately than total gate count;
reject if bootstrap rank-error intervals overlap or improvements vanish across
seeds. Benchmark a constant predictor too.

**Minimum/baselines.** 4–6 spins, two times and two observables, 3–5 feasible
layouts, five compiler seeds; compare count, depth, calibration product and
causal-cone score against exact references and simulated noise. Hold out times
and entire layouts. Match shots and charge calibration if hardware is used.

**Knowledge/resources/hardware.** Add circuit graphs, physical/logical mapping,
Spearman correlation, bootstrap and cross-validation. Fake-backend local testing
suffices initially; actual layout superiority requires hardware.

**Risks/negative result.** Noise models derived from the same calibration score
make the test circular; routing swaps, drift, mapping mistakes, correlated
seeds and tiny samples confound ranking. Finding all metrics equally weak is
useful if uncertainty is explicit.

**Two weeks / 8–12 weeks.** Reproduce fixed-layout compilation and verify
observable remapping locally; subsequently evaluate noise families, held-out
rankings and optionally interleaved hardware repeats. Do not turn compiler-seed
search into a claim of new optimization.

## 3. Predicting target dynamics with mirror diagnostics

**Question/motivation.** Can normalized Hamming return or all-zero return of a
matched mirror predict Z0 dynamics error after controlling for depth and gate
count, on circuits it was not trained on?

**Closest work/challenge.** C23 explicitly compares mirrors and unmirrored
mitigation and finds optimistic behavior. The general proposal is already
studied. Newer stabilized-noise hardware work also uses mirrors; a fresh search
is required before any narrow novelty claim.

**Unresolved possibility/contribution.** Evaluate prediction of a specific
observable using held-out fields/times and coherent-noise stress tests, rather
than just plotting mirror success versus target success. This could be an
incremental robustness dataset, subject to redundancy review.

**Hypothesis.** Mirror metrics reduce held-out prediction error beyond count/depth
baselines. Falsify with no improvement or failure under coherent mismatch.

**Minimum/baselines.** 4–6 spins, 20 matched circuit pairs, two observables and
three noise families; exact density/state calculations. Compare constant,
gate-count and per-qubit return baselines. Cost includes diagnostics as well as
target shots; match physical qubits and native operations where possible.

**Knowledge/resources/hardware.** Linear regression, controls, held-out testing,
coherent noise and circuit inversion in addition to candidate 1. Local model
study possible; hardware transfer is optional but substantially stronger evidence.

**Risks/negative result.** Inverse cancels coherent errors, readout dominates
all-zero return, circuit pairing changes causal cones, train/test leakage and
vanishing return at large n. A well-isolated counterexample is useful even if
prediction fails.

**Two weeks / 8–12 weeks.** Reproduce one C23-style optimistic mirror example;
then evaluate held-out transfer with matched resources and topology. Do not
promise an error certificate from empirical correlation.

## 4. Noise-scaling mismatch and temporal robustness

**Question/motivation.** How much miscalibration in intended noise gain changes
the fixed-budget choice between raw and extrapolated estimates for a spin
observable? Mitigation depends on a controllable noise model.

**Closest work/challenge.** K23 learns noise and discusses stability; C23 shows
simplified models can fail. Z26 assumes achievable amplification above an
intrinsic floor. The general issue is established.

**Unresolved possibility/contribution.** A finite-budget sensitivity map for
specific observable/circuit choices, including a conservative “do not mitigate”
decision. Need a targeted review of robust/adaptive ZNE before novelty claims.

**Hypothesis/evaluation.** A method chosen under perfect scaling loses a
predeclared >=20% MSE benefit under modest gain mismatch or drift; quantify the
threshold rather than selecting only successful examples.

**Minimum/baselines.** Extend current density-matrix sweep with scale errors
±5/10/20%, damping and coherent overrotation, three temporal blocks. Compare
raw, nominal linear ZNE, and gain-aware oracle correction; the latter is an
upper benchmark and must not be advertised as practical. Charge calibration.

**Knowledge/resources/hardware.** Channels, model misspecification, paired
sampling and bootstrap; local laptop sufficient. Real drift claims require
repeated hardware data, unavailable here.

**Risks/negative result.** Arbitrary mismatch model, changing multiple knobs at
once, oracle gains and shot-limited calibration. Robust performance across the
predeclared range is a useful negative result against the fragility hypothesis.

**Two weeks / 8–12 weeks.** Run deterministic gain mismatch first; later add
calibration uncertainty and blocked chronological validation, subject to review.

## 5. Classical resource comparisons for a declared observable

**Question/motivation.** For chain dynamics, when are exact exponentiation,
statevectors, MPS or a local causal-cone method the cheapest verified route to
Z0 at a fixed accuracy? This directly addresses misleading statevector-only
resource claims.

**Closest work/challenge.** T24 and BGC24 demonstrate why representation matters;
tensor-network and integrable Ising methods are mature. Generic memory tables
are a learning exercise.

**Unresolved possibility/contribution.** A reproducible accuracy/time/memory
comparison for the exact tested circuits, with convergence and integrability
baselines. A new simulation algorithm is out of scope.

**Objective.** Compare time-to-additive-error .01 on small exactly verifiable
cases, then scale only methods with convergence controls. Do not infer hardness
from any timeout.

**Minimum/baselines.** 4/8/12-spin short circuits, exact small references, Aer
statevector/MPS, truncated causal cones where justified; transverse-field
free-fermion treatment as an important further baseline. Report threads, setup
costs and memory, not just kernel time.

**Knowledge/resources/hardware.** Linear algebra plus Schmidt decomposition and
algorithm analysis; laptop, bounded runs; hardware unnecessary.

**Risks/negative result.** Unfair implementations, unreported MPS truncation,
memory allocation versus actual use, and a trivial integrable model. If simpler
methods dominate, that is an informative baseline result and a reason to avoid
an advantage claim.

**Two weeks / 8–12 weeks.** Reproduce a small scaling comparison; subsequently
validate convergence and compare a supervisor-selected model with longitudinal
field or a different observable if scientifically motivated.
