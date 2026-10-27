# Active focus: QNNs and quantum attention

Persistent experimental records are in `logs/`. Read prior `*_result.json` and
`09_checkpoint.json` before proposing another experiment; preserve earlier logs.
Each cycle writes a plan before execution and a result/decision afterward. The
code refuses to overwrite named logs. This directory supersedes classical memory
as the current focus; earlier results remain separate and unchanged.

## Current implementations

- `core.py`: two-qubit Qiskit QNN and overlap-attention circuits; vectorized
  state simulation independently checked against Qiskit. Classical softmax/readout.
- `experiments.py`: trained variational QNN, ideal and finite-shot tests, matched
  classical feature control. Disjoint train/validation/test inputs.
- `attention.py 2`: cancelled-parameter diagnostic and compute/uncompute check.
- `attention.py 3`: train initial interleaved quantum feature map.
- `attention.py 4`: consistently scale every data re-upload; new data seeds.
- `shots.py`: fixed-shot uniform vs adaptive top-one readout, fresh episodes.
- `noise.py`: actual Aer density-matrix simulations with modeled gate noise.
- `noisy_shots.py`: finite-shot readout from noisy circuit probabilities.
- `value_attention.py`: evaluate softmax attention @ values, sparse approximation,
  and validation-only temperature calibration with its overhead stated.

This is a trained QNN and a transformer-related quantum attention component,
not a full language transformer. All computations are local simulations. Small
circuits are classically simulable. Training uses exact simulated expectations;
finite-shot training itself has not been tested. Adaptive allocation is classical
control of quantum measurements, not a quantum-only algorithm.

Run from the project root using the existing environment:

```sh
.venv/bin/python research/quantum_nn/continue.py --logs /tmp/qnn-new-replay
```

Use a new directory; this replays frozen history rather than autonomously selecting
new hypotheses. Source hashes and dependencies are in `logs/provenance.json`.
Do not infer energy savings, hardware runtime or quantum advantage. No hardware,
credentials, packages or paid compute used. Current useful direction: the effect
of observable parameter placement and finite-shot allocation on quantum attention.
Generic adaptive acquisition is established work; novelty remains unverified.

## Continuation: actual small hybrid block

Latest checkpoint: `logs/15_checkpoint.json`. New results10–14 follow checkpoint09
(the checkpoint is not experiment9). There are now13 experimental cycles total.

- `robustness.py`: frozen-map candidate scaling4–32 and domain shift.
- `encoding_range.py`: encoding frequency/range/shot-resolution tradeoff.
- `hybrid_block.py`: complete minimal hybrid cross-attention block,50 trainable
  parameters, fixed projections, residuals, normalization and trained FFN/readout;
  classical metric head control has the same parameter count.
- `block_readout.py`: frozen-block dense versus adaptive sparse readout on fresh data.
- `validate_block.py`: exact input collision, independent circuit check and
  candidate permutation invariance.

This is a small hybrid cross-attention Transformer-style block, not a standard
full Transformer implementation or an LLM. Both heads warm-start from earlier
retrieval training. Two quantum fits hit the60-iteration limit; validation-selected
checkpoints are provisional. Full optimization traces/counts/messages are saved.
Adaptive sparse and uniform dense attention differ in semantics and feedback
rounds. Equal shots does not imply equal wall time. Experiment11 contains repeated
scale*frequency products by design; those conditions are not independent evidence.
The replay runner includes these five follow-ups. It generates result/plan logs,
not the manually written checkpoint/provenance files. `review()` now ignores
metadata files when selecting the most recent substantive log.

## Capability/generalization/reliability focus

Read `logs/24_checkpoint.json` first. The user changed the target from efficiency
to learning capability, reliability and breadth. Eight more cycles16–23 test:

- `capabilities.py 16`: basic QNN and classical controls,4 tasks,3 sample sizes.
- `capabilities.py 17`: repeated data encoding, same tasks/data and optimizer cap.
- `matched_control.py`: exact classical function-family control of the basic QNN.
- `expressivity_check.py`: diagnose the re-uploaded output's missing frequency sector.
- `qnn_mixture.py`: learned two-circuit mixture,9 retained trainable parameters.
- `reliability.py`: calibrated probabilities, selective prediction, corruption,
  and finite measurements split across both quantum circuits.
- `compositional.py`: completely withhold one input quadrant, no held-out-region
  labels in training/validation/checkpoint selection.
- `generalization_control.py`: additional generic Fourier control on that shift.

Total21 experimental cycles (checkpoint numbers9/15 are not experiments).
All are small synthetic studies, not general intelligence or a published research
gap. Circuit-friendly tasks and architecture-informed features are explicit.
Repeated-data QNN is not automatically more expressive; selected means from3
seeds are exploratory. Nested sample sizes/same datasets across architectures
are paired, not independent trials. Generic baseline hyperparameters are not
exhaustively tuned. Same parameter count does not mean the same inductive bias.

The basic QNN logit has an exact3-parameter classical representation, verified
on independent points. The mixture combines two measured circuits classically;
it does not introduce an untested quantum coherent ensemble. Its confidence
calibration is fitted on clean validation data and is not a guarantee under shift.
Raw confidence coverage at.9 and calibrated validation-selected coverage follow
different rules; do not attribute their difference solely to temperature scaling.
Numeric replay runner includes all these cycles; sources/results are preserved
by write-once log names. See provenance_24.json for updated hashes/environment.

## Multi-output quantum models and explicit uncertainty

Latest log: `logs/34_checkpoint.json`. Nine follow-up cycles25–33 bring the branch
to30 experimental cycles. Read the checkpoint and substantive results before new
experiments; preserve every earlier log.

- `feature_readout.py 25/26`: fixed two-qubit feature encoders, multiple Pauli
  readouts, single scale or frequency bank, learned classical output weights.
  These are hybrid quantum-feature models, not end-to-end trained quantum gates.
- `measured_features.py 27/28`: physically shared outcome counts for full9-basis
  and reduced4-basis measurements. All view/basis shots charged exactly; covariance
  within a setting preserved. Reducing settings does not help every task.
- `select_features.py`: validation-only model selection, fresh-data comparison.
- `prediction_sets.py`: independent uncertainty calibration, correct/ambiguous/
  empty-answer bookkeeping; marginal coverage differs from single-answer accuracy.
- `corruption_calibration.py`: representative noisy calibration, ambiguity cost.
- `feature_control.py`: exact elementary classical representation of these small
  fixed circuit features. No uniquely quantum capability demonstrated.
- `new_tasks.py`: frozen encoding scales challenged on four additional task families.
- `validate_feature_measurements.py`: independent estimator/reconstruction checks.

The same encoding family can support broader toy tasks by retaining multiple
outputs and scales, but calibration and measurement overhead matter. Model choices
are made per supervised task, not by one task-agnostic universal QNN. New diagonal
wave task remains a major failure, motivating learned input directions/scales.
All new finite-shot tests exclude gate noise; earlier gate-noise results do not
validate this different measurement/readout architecture.

Prediction sets can achieve about90% marginal coverage on represented corruption
by including both labels often. They do not magically restore classification or
certify individual answers; model-selection validation is never reused as the
new conformal calibration set. Source/result hashes: provenance_34.json.

## Learned input encoding continuation

Latest substantive entry: `logs/40_checkpoint.json`. Five cycles35–39 bring the
branch to35 experimental cycles. Model inputs are now mixed by a learned2x2
matrix before the actual RY gate angles, jointly trained with observable-readout
weights. This adds4 parameters and more optimization; it remains exactly
classically reproducible at this small size.

- `learned_inputs.py`: paired frozen-vs-learned encoders on new cohorts; analytic
  gradient independently checked. Two starts/config,200 iterations.
- `encoding_confirmation.py`: locked protocol on four new wave directions, with
  classical13/37-parameter MLPs and a Fourier control.
- `encoder_optimization.py`: targeted400-iteration continuation with training/
  validation traces and new test data. Ten targeted runs converged; this does not
  erase iteration-limit failures in other candidate runs.
- `learned_measurements.py`: frozen checkpoints, correct grouped1024/4096-shot
  measurement,all views/settings counted; no gate noise.
- `encoding_extrapolation.py`: enlarged input domain,unchanged analytic labels;
  reports balanced accuracy to expose label imbalance.
- `validate_learned_inputs.py`: Qiskit/gradient validation.

Learned angles improve the diagonal-wave problem substantially and preserve its
periodic relation farther outside training support. This is a task-specific prior,
not proof of broad reasoning. Other extrapolation tasks still fail. Larger-model
or hardware conclusions require new evidence. Replay includes this continuation;
use a fresh log directory. See provenance_40.json for source/result hashes.

Latest continuation: `logs/44_checkpoint.json` (38 experimental cycles).
`nonwave_confirmation.py` tests four non-wave boundaries with classical controls.
`noise_aware_readout.py` refits frozen encoders using grouped measured features;
`noise_confirmation.py` checks the ring reliability lead on fresh data.
At256 shots, confirmed confidently-wrong fraction falls7.58% to2.28%,
while accuracy changes85.27% to85.70%. At1024 shots accuracy slightly worsens.
Three training seeds only; no gate noise or unique quantum advantage.
Replay includes all three cycles. Source/result hashes: `logs/provenance_44.json`.

2026-10-27: latest `logs/49_checkpoint.json`,42 experimental cycles.
New scripts: `independent_noise_training.py` (5 independent training seeds),
`confidence_control.py` (positive-temperature baseline),
`device_noise_transfer.py` (Qiskit-validated toy depolarization/readout channels),
`equal_shot_grouping.py` (same measurement records regrouped1/2/4 ways).
Low-shot ring accuracy84.4%->87.6% with noisy readout training;temperature
explains much of confidence improvement but cannot alter labels.
At1024shots temperature is competitive/better. Grouping uses identical total
training shots;known related preprint arxiv2604.28160 means no novelty claim.
See checkpoint for sources,scope,next tests and candidate professor question.

Latest: `logs/53_checkpoint.json` (45 experimental cycles).
`grouping_breadth.py` tests all4 non-wave tasks at equal grouping measurement
budgets,with cheaper temperature controls. `affine_control.py` adds a2parameter
scale/intercept baseline. `asymmetric_drift.py` compares stale versus refreshed
readout selection in two hypothetical asymmetric channels. Grouping is not
universally helpful;matched recalibration improvesmean CE across4tasks,but
stale correction can be worse than no device-specific correction. No drift
detector orIBM-calibrated channel yet. Candidate direction:monitoring/calibration
cost under changingmeasurement errors. Sources andlimitations in checkpoint.

Latest `logs/58_checkpoint.json`:49exploratorycycles.
`reference_monitor.py`:8referencegroups,chargedbaseline+monitoring,bootstrap
changealerts. `triggered_recalibration.py`:independent64calibrationlabels,
never/always/triggered policies. `randomized_monitoring.py`:newtimings/magnitudes,
periodic3/6/8controls. `confidence_guard.py`:freshtrajectorytest offixed
logitcap4. Monitoringcanhelpaccuracy/cost butprobabilityerrorssometimesworsen;
caphelpsCEwhileBrierisslightlyworse. No universalreliabilityclaim.
Next:regularized/heldoutcalibration,freshtrainingseeds,unseennoisemechanisms.
Source/result hashes in `logs/provenance_58.json`;replayincludes4newrunners.

Publishedto https://github.com/Art200507/Quantum-Research;initialcommit82d7ff5.
Researchcontinuedafterpublication:59heldoutregularizedcalibration,60nested
calibrationbudget64/128/256withsharedtestrecords. Latest61_checkpoint.json;
51exploratorycycles. LargerbudgetimprovesmeanMLECE .279->.197;caphelpssmall
budgetbuthurts256labelcase. Regularizationnotconsistentlybetter. Both128label
capbatchesdiffer;keepuncertainty. RootREADMEexplainsfindings/aims/reproduction.
Next:temporaldeploymentconfirmation,freshtrainingseeds,andunseenchannels.

Latest65_checkpoint.json:54exploratorycycles.62fresh_encoders.py retrains
3newseeds/task;63temporal_budget.py pairs64/256labelupdateswithsharedtest
records,removesalwaysbatch0duplicate. Freshmodeltestdoesnotconfirmmonitor
superiority:256labelneverCE.20544/262144shotsvstriggered.20586/348160.
64monitor_blindspot.py:YY-onlyfaultinvisibletoZZ/XXreferencepanel;expanded
paneldetectionweakandactualpredictionimpacttiny. Thisisnotademonstrated
harmfulblindspot. NullresultsinrootREADME;nextprioritizeprediction-relevant
referencepanelsandstrongnever/periodiccostcontrols. Replayincludes3runners.
