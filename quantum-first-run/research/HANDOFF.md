# Handoff — 25 October 2026

## User constraints and background

UB undergraduate, intermediate Python, no advanced math/quantum courses reported.
Weekly availability unanswered; 6–8 hours/week is a planning assumption. Seek a
credible question and preliminary evidence for approaching a professor, not a
paper or advantage claim. Autonomous reversible local work authorized; no hardware
jobs, paid workloads or credential access. Preserve explicit hardware confirmation.

## Completed

Inspected all project sources, pins, saved JSON, local installed runtime Sampler
source/API and environment. No Git repository or applicable AGENTS.md found.
Read-only credential paths were never accessed. Existing historical results kept.
Copies of original modified files saved in research/originals/. Corrected misleading
mirror/readme claims and plotting labels, QRNG multiple-test flags, output-directory
handling and basic invalid CLI arguments. Hardware confirmation gate retained.

Live primary-source search found substantial prior overlap: E19 optimal Trotter
steps, C23 mirror/mitigation benchmarking, S26 fixed-shot MSE and Z26 joint
mitigation. No novelty established. K23 full text ultimately accessed at PMC
because Nature links failed. Access depth/code availability are recorded honestly
in LITERATURE.md. Five candidate questions, recommendation/fallback, roadmap,
actual results, professor note, direct dependency pins and bibliography written.

Pilot: 3/4-spin open chains, times .5/1/2, first/second-order formulas, 1/2/4/8/16
steps, p=0/.001/.005/.02, Z0/Z0Z1, budgets 200/2000/20000. Oracle p-to-3p
amplification, raw/equal-split/3:1 ZNE. Generated 4320 estimator rows, 2000
synthetic trials each, seed 20261025. Seven validation checks passed; state error
ratios on halving step size 2.00102795/4.01333348. No sampled-MSE discrepancy
beyond six Monte Carlo SE. Numerical sweep runtime 1.56 seconds excluding
plotting, with larger environment/cache initialization overhead. Largest density
array 4096 bytes; no large compute. analyze.py generated equal_work.csv and
RESULTS.md; figure visually inspected. Model and limits are explicit.

Compiler audit: seed_transpiler=17. With midpoint barrier, CZ counts 32/152/392/792
for n=5/20/50/100; levels 3/3/1/1; depths 59/76/81/84. Without barrier n=5/20
at level 3 has zero two-qubit gates. Exact all-to-all 5-qubit compilation confirms
ideal return 1 with/without barrier. No broad equivalence or compiler-seed study.

Local 128-shot demo passed for QRNG, Bell, GHZ and n=3 mirror using FakeTorino.
Final output directory contains JSON and three PNGs; mirror return .8984375,
Bell support .9921875, GHZ support .96875. Noisy counts are unseeded and can vary.
This smoke test validates pipeline operation, not scientific claims.

## Exact reproduction commands

Working directory `/Users/AtharvaBadgujar/Quantum/quantum-first-run`:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantum-mpl .venv/bin/python research/pilot.py
.venv/bin/python research/analyze.py
.venv/bin/python research/checks/audit_local.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantum-mpl .venv/bin/python run.py --shots 128 --mirror-sizes 3 --out research/results/demo_check
```

The first actual pilot invocation from the parent workspace redirected stdout to
research/results/pilot_stdout.json; compiler stdout to compile_stdout.json. The
original demo smoke invocation aborted with exit 134 during GUI plotting. Retrying
with MPLBACKEND=Agg passed (exit 0); matplotlib/fontconfig emitted cache warnings.
No claim of a numerical failure is inferred from the plotting abort. Temporary
edit helper initially failed before any edits due to Path/string concatenation;
fixed and applied once. Originals were copied before edits.

## Findings to communicate

At n4,t1,order2,p.005,Z0,8 steps, ZNE 3:1 lowers absolute bias .01260 -> .004071
but at B200 increases MSE .004392 -> .01707. At B20000 it slightly beats raw at
the same depth; raw 4-step MSE .0001692 beats best tested 3:1 ZNE .0001825.
Thus a tuned raw baseline changes the interpretation. This is replication/model
illustration, not novelty. Noise amplification and gate-work models are idealized.

## Unresolved / next action

User should independently reproduce and explain one result while learning vector
algebra/probability; then discuss the concept note with a supervisor. Jamir Marino
and Xuedong Hu were identified through current official UB pages as plausible
scientific contacts, with no claim of availability and no contact made.

For research continuation: inspect full E19 and S26/Z26 code; complete forward
citation/robust-ZNE/layout literature search; test actual folding with reoptimization
controlled, native compilation, coherent/damping/readout noise and sensitivity to
field/initial state. Any practical method selection needs charged training and
held-out validation. No implemented result yet establishes a publishable gap.
No executable hardware plan is justified. Native compiled pilot sizes, calibration
costs and duration estimates remain undone intentionally pending evidence.

## Update — 26 October 2026: memory/AI investigation

User redirected discussion toward quantum-inspired efficient neural architectures,
then explicitly authorized ten sequential experiment/review/refinement cycles.
Completed all ten in `research/memory_lab/`, preserving earlier code and outputs.
See its README, REPORT, JOURNAL, SOURCES and results manifest for exact conditions,
commands, measured costs and next action. No subagents, packages, hardware, paid
workloads or credentials used. 2,688 estimator rows plus 672 verified three-qubit
circuits; all results local. Experiment 10 uses fresh seeds 1000–1023.

Outcome: single-vector phase binding fails under load; equal-byte real control
shows no consistent phase advantage. Classical linear/ridge baselines dominate.
Int8 ridge retains 100% recall for the sampled 256-pair/mild-noise regime with
4,100-byte state versus 66,564 bytes compressed nearest-key payload; larger loads
and stronger noise expose substantial failures. Dynamic corrected int8 updates
reach 99.93% on exploratory seeds with 4,612 bytes including old-label metadata.
Quantum overlap encoding verified numerically but loses signed information and
adds preparation/readout costs. No QNN training, transformer superiority, novelty,
100x speedup or power saving established. Timings are narrow warmed microbenchmarks.

Replay utility protects original numerical outputs. New-directory experiment-1
replay matches original summaries. Python syntax checks passed; generated figure
visually inspected. Initial replay-check helper incorrectly resolved the venv
interpreter symlink to global Python and hit missing NumPy; using sys.executable
fixed this. No experiment data was affected. All authorized ten cycles are done.

Next research direction if continued: structured associative-recall benchmark,
absent/conflicting keys and mixed updates/queries, task-matched modern baselines,
full encoding/workspace cost, and closer prior-art review. Do not treat a synthetic
memory payload comparison as a general neural-network improvement.

## Update — 26 October 2026: user correction to actual QNN/quantum-transformer focus

User explicitly asked to focus on quantum computing/QNNs/quantum transformers,
continue experiments until useful, and keep persistent logs rather than a report.
Completed eight new sequential local experiments in `research/quantum_nn/`.
Read that folder's `logs/09_checkpoint.json`, latest result JSONs and README
before any future experiment. `logs/provenance.json` records source/result hashes
and versions. No classical-memory claim should be substituted for this focus.

Implemented actual two-qubit Qiskit variational QNN and trained fidelity-attention
circuits with a checked vectorized simulator. All five QNN training runs converged;
ideal test accuracy 97.23%, matched classical features 96.99%, raw linear50.08%.
Diagnosed shared-final-unitary cancellation and one unobservable QNN parameter.
Initial interleaved quantum attention 85.16%, consistently scaled re-upload model
95.55% on fresh cohorts, learned classical metric98.40%. These are toy tasks,
not quantum advantage or full-transformer results. Training uses exact expectations.

Fair-shot adaptive top-one attention beats uniform on independent episodes;
under actual modeled noisy circuit probabilities, B512 gives 77.65% uniform
versus82.94% adaptive (5 fitted seeds x128 fresh episodes). Weighted-value output
experiment also shows reduced MSE but sparse and dense semantics differ and
calibration overhead is explicit. Circuit equivalence, density trace, and shot
accounting checks passed. No hardware, credentials, paid workloads or installs.
Useful stepping stone achieved: quantum-attention architecture/readout baseline,
not novelty, hardware efficiency or superior language model. Prior AQKA/QASA
abstracts show directly relevant established work; full-code inspection remains.
Next experiment priorities and exact replay entry point are in quantum_nn logs
and README. No experiments are running after this turn.

## Continuation — quantum-attention scaling and complete hybrid block

Completed five follow-up cycles (results10–14; checkpoint09 was not an experiment).
Latest entry point: `quantum_nn/logs/15_checkpoint.json`; current source hashes
in provenance_15.json, preserving the original provenance. Total13 experimental
cycles in this branch. Read latest substantive logs before new work; core.review
now filters metadata instead of accidentally selecting provenance.json.

Added generalized power-of-two allocation, candidate/domain-shift study,
frequency/range tradeoff, a50-parameter hybrid cross-attention block with residuals,
fixed LayerNorm and a jointly trained FFN/readout, matched-count classical block,
fresh-data sparse/dense deployment study, and exact structural tests. No packages,
credentials or hardware used. All scripts pass syntax checks; vector/circuit,
permutation, exact collision and shot-bound checks passed.

Frozen-block deployment on three fresh256-episode cohorts: quantum ideal97.40%,
uniform512shots91.37%, adaptive512shots93.85%, classical98.96%. Sparse and dense
semantics differ; shot allocation is not a latency/energy claim. Two quantum
training fits hit60 iterations, and validation-selected checkpoints are provisional.
Warm-started heads and fixed token projections mean this is a minimal hybrid
cross-attention Transformer-style block, not a full standard Transformer or LLM.

Range-shift failure is severe. Exact periodic input collision verified by Qiskit;
smaller frequencies can recover ideal ranking but shrink shot-resolvable gaps.
New logs preserve all grid conditions, not a chosen winner. Inspect QASA/AQKA
full-text notes in checkpoint15 before novelty claims. Future experiments should
address range/encoding and optimization convergence rather than scaling blindly.
Replay utility includes follow-ups, always use a new log directory. No jobs remain
running at the end of this turn.

## Capability/generalization/reliability continuation

User clarified capability, reliability and task breadth are the objective, not
speed, and requested experiment/review/log cycles. Completed eight additional
cycles16–23. Latest starting point `quantum_nn/logs/24_checkpoint.json` and
provenance_24.json. Branch now has21 experimental cycles total; checkpoints are
not experiments. Review latest logs before new work. No work remains running.

New suite:4 tasks(linear,cosine interaction,radial,mixed frequency),3 training
sizes16/64/128,3 independent data seeds,128 validation/512 test. Same nested data
across models; basicQNN, re-uploadedQNN, raw linear,9/33-parameterMLPs, generic
Fourier and exact architecture-feature control. QNN circuit checks passed. Basic
QNN cosine skill98% is exactly classically representable (logit error1.07e-14).
Re-uploading improves other tasks but loses cosine feature sector; Fourier
numerical diagnostic finds odd/odd sector absent. Learned two-circuit mixture
retains cosine97.7% and increases linear80.5% but radial71.1%,mixed61.2%.
No broad-capability or quantum-advantage conclusion.

Reliability calibration/abstention and total256/1024shot tests logged. Clean-data
calibration can fail under input corruption; empirical90% validation selection
is not a guarantee. Structured quadrant holdout: mixture cosine98.0% vs rawMLPs
52.8/64.8%; matched classical feature model98.7%. Radial69.3%vsMLPs~52%,generic
Fourier66.0%, small3-seed study; mixed remains~51.9% and strongerMLP~72.9%.
All baselines and negative results retained; no exhaustive hyperparameter tuning.
59/288 initial optimization runs hit120-iteration limit; later re-upload and
mixture runs converged. Numerical checks passed: Qiskit mixture branch error
5.56e-16, baseline gradient1.14e-11, disjoint sample checks, syntax checks.

Added scripts capabilities.py,matched_control.py,expressivity_check.py,
qnn_mixture.py,reliability.py,compositional.py,generalization_control.py,
validate_capabilities.py; replay includes them. No packages/credentials/hardware
used. Next: more independent cohorts/validation-only tuning, preserving useful
quantum feature sectors, and calibration evaluated on held-out corruption families.

## Multi-output capability and uncertainty continuation

User requested continuation and learning from each experiment. Completed nine new
cycles25–33, with plan/result logs reviewed sequentially. Branch now30 cycles.
Read `quantum_nn/logs/34_checkpoint.json` first; provenance_34.json preserves
current sources and result hashes. Previous logs/results unchanged. No jobs remain
running. No packages, credentials, paid work or hardware used.

Added multi-Pauli readout to fixed2-qubit quantum encoders, first single frequency,
then4 frequencies. Learned classical readout is a hybrid quantum-feature model,
not newly trained variational gate parameters. Single encoder has8 nonzero retained
features; bank32; output heads9/33parameters. Actual Qiskit feature verification
passed. Generic classical comparators and exact elementary feature representation
included; features/logits agree within4.45e-16/3.56e-14, so no unique quantum claim.

Fresh5-seed original-task validation-selected models: ideal94–98%,1024-shot93–97%.
Grouped categorical measurement simulation correctly shares counts/covariances;
all view/basis shots charged.4-basis restriction covers retained features but does
not uniformly improve estimation over9bases. Calibration uses new data separate
from model-selection validation. Representative noisy calibration restores~90%
prediction-set coverage, but on mixed-frequency corruption~71% outputs ambiguous
both-label sets and classifier accuracy remains61%. No conditional reliability
or intelligence improvement inferred from coverage.

Further four task families without changing encoder frequencies: rotatedellipse
~97% at1024 shots,diagonalwave~68%vs selectedclassical96%,annulus88%,tiltedinteraction
82%. Fixed input directions/scales are a clear next limitation. Potential next
step: learn input mixing while preserving rich readouts, using new untouched task
cohorts and matched classical controls. Do not misstate per-task model selection
as one task-agnostic model or inherit old gate-noise results for these new readouts.

New validation checks passed: exact basis reconstruction,circuit/elementaryfeature
agreement,shot budgets,MonteCarlo marginal means,prediction-set bookkeeping,syntax.
One initial validation used empiricalSE and falsely failed a nearly deterministic
observable (zero rare flips in500 repeats); corrected test uses known Born marginal
variance. Measurement estimator and numerical experiment outputs were unchanged.
Replay utility/README include all cycles; use a new log directory for replay.

## Learned input encoding continuation

User requested continuation. Completed five cycles35–39 after reading checkpoint34.
Branch now35 cycles. Latest: quantum_nn/logs/40_checkpoint.json,provenance_40.json.
No jobs remain running. Logs immutable/preserved; replay includes continuation.

Jointly learn a2x2 input-mixing matrix and quantum-observable readout;actual circuit
RY gate angles f*(x@W),CX,multiple measured observables. Same2-qubit family remains
exactly classically reproducible;no quantum advantage/new algorithm claim.
Analytic simulator gradients independently checked,learned circuits verified by
Qiskit (gradient1.58e-10,feature derivatives4.8e-10,circuit1.50e-15 maxerrors).

Paired fresh5-seed tests: diagonal ideal78.6%fixed->97.2%learned;other tasks mostly
retained,tilted modest improvement. Four new wave directions,3seeds:learned94–99%,
fixed71–92%;classical controls competitive.44/80 initial encoder runs and16/48
confirmation runs hit200 iterations. Targeted400-step continuation converged10/10,
with only modest test improvement. Do not erase other failures.

Frozen-model finite-shot tests: diagonal1024shots69.9%fixed vs96.9%learned,new
512-point cohorts. Gate noise absent. At3x input range,ideal diagonal94.7%learned
vs54.1%fixed;other tasks still degrade. Balanced/majority metrics saved for radial
label imbalance. Wave family is deliberately favorable to learned directions;
no universal intelligence or task-agnostic model claim. Per-task training.

Added learned_inputs.py,encoding_confirmation.py,encoder_optimization.py,
learned_measurements.py,encoding_extrapolation.py,validate_learned_inputs.py.
Syntax checks passed,no credentials/hardware/paid work/installs used. Next: new
non-wave task families with frozen protocol,noise-aware training/calibration,and
continued exact-classical/neural controls before any hardware proposal.

2026-10-26 continuation41–43: read preceding logs before each run. Latest44_checkpoint.json;38 experimental cycles. New non-wave tasks: learned encoder improves parabola/XOR, slightly hurts ellipse/ring;20/48 encoder runs hit cap. Noise-aware frozen-encoder readout:96/96 optimizations converged. Fresh ring confirmation256shots:CE .506->.328,confidently wrong7.58%->2.28%,accuracy85.27%->85.70%;1024shots accuracy91.08%->90.58%. Three original training seeds, fresh confirmation data,10 paired measurement repeats; repeats not independent seeds. Follow-up selected after42, not preregistered independent task replication. No gate noise/hardware/quantum advantage. Next: new independent training seeds with frozen protocol,then justified gate-noise/control experiments. Added3 scripts,replay updated,Qiskit/gradient validation passed;provenance_44 preserves hashes.

2026-10-27 cycles45–48 complete;latest quantum_nn/logs/49_checkpoint.json;42 experimental cycles. Reviewed preceding logs before each. Independent5seeds10..14 ring256shots:accuracy84.42->87.65%,CE .556->.302,confidently-wrong8.42->2.27%;CE improves all5. 10/20 encoder runs converge,all retained.46 fresh-test temperature control explains much reliability gain without changing labels;noise improves low-shot decisions,temperature better high-shot.47 toy global depolarization+readout flips:relative gains survive,all accuracy degrades;Qiskit density validation5.55e-16.48 same1024training shots/point regrouped1/2/4:4groups best mean accuracy;validation chooses4for4seeds/2for1. Exact45 weights reproduced. Gradient/circuit checks pass;replay includes4scripts. Related primary abstracts read:Guo17 temperature,arxiv2306.01639 variance regularization,2604.28160 fixed-budget grouping,2410.14654 noise theory. Methods already studied;no novel/quantum advantage claim. Next:equal-cost broader tasks/channel-aware and drift controls;full-paper reading before professor/IBM proposal. No hardware/credentials/paid work.

2026-10-27 continuation50–52 complete;latest53_checkpoint.json;45experimental cycles.50groups across4tasks:temperature dominates parabola/XOR,oftenellipse;ringG2 helps.51affine(scale+bias) stronger cheapbaseline doesnot erase modestring gains;12/12converge.52hypothetical asymmetric readout A/B plusglobaldepol,device-matched pipeline improvesCE inall4tasks/bothenv means;staleA afterB canhurt(parabola86.24% vs shot-only92.83% vsrefreshB95.53%). XOR alwaysselectsaffine. Uses3existing41encoders/task,newmeasurement/testdata,notindependentretraining. RestoredvalidationrecordCEexact;Qiskitgradient/channels pass;allnewsyntaxpass. Replayed3scripts added;provenance53. Readarxiv2604.28160v2method/setup:knownshotgrouping,exactcovidentityforsquaredloss notourlogisticreadout. Next:independentcalibrationlabels/budgets,reference-circuit driftmonitor withchargedshots,freshindependenttrainingcohorts. No detector,realtemporaldrift,IBMcircuitcalibration,noveltyorquantumadvantage claim. Localonly,nohardware/credentials/paidjobs.

2026-10-27 cycles54–57complete;latest58_checkpoint.json;49exploratorycycles.54monitor512shots:fullA->B detection99.85%,falsealarm.95%,quarter9.5%/half56.6%;onebaseline,eventratesonly.55triggeredabruptmatchesalwaysaccuracyat40%fewertotalshots,butstationaryoverhead/gradualdelay.56newrandomtimings/magnitudesplusperiodic3/6/8:triggered91.148%/254976shotsvsperiodic6 90.938%/262144shots;CEworse(.2504vs.2460),abruptXORoutliers.57freshsequencesfixedcap4:triggeredCE .2360->.2199,accuracy90.777%/265216shotsunchanged;Brierslightlyworse.150/3456affineboundhits,maxrawlogit693.6;allupdatesconverge. Fresh64calibrationlabelsdisjointencoderselection/test,reusedupdates;old12encoders. Monitorreferencebaseline4096+512/check+4096/alarmrefresh,adapt16384/alarm;costsverified. SciPyGstaterror2.84e-14;Qiskit/gradientchecksandsyntaxpass. Replayupdated. Nextconsolidatedbenchmarkwithregularized/heldoutcalibration,freshtrainingseeds,unseenmechanisms/familywisealarmtests. Alwaysincludesextrabatch0refreshafterinitialcal;skipnextbenchmark. Fixedcapnotcalibrationguarantee. Localonly,nohardware/credentials/paidjobs.
