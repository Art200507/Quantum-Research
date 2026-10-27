# Quantum Research

An ongoing, AI-assisted experimental project investigating small quantum neural networks (QNNs), hybrid quantum/classical learning, and reliable predictions from limited quantum measurements.

Our aim is to develop a reproducible research question that can be discussed with university researchers and, when justified, tested on IBM quantum hardware. We care about capability, generalization, and reliability—not just speed.

**Current status:** 51 logged exploratory experimental cycles through [checkpoint 61](quantum-first-run/research/quantum_nn/logs/61_checkpoint.json). These are iterative toy studies, not 49 independent replications. Research continues after this repository snapshot.

## What we are trying to find

The broad motivation is to understand whether quantum components can contribute useful learning behavior. The most concrete question emerging from our experiments is:

> Can a small set of reference measurements detect changing device errors and trigger safe recalibration of a hybrid quantum classifier, with a better tradeoff between prediction quality and measurement cost than scheduled updates?

A QNN here is a small quantum circuit that transforms numerical inputs into measured features, followed by a trained classical prediction layer. Some experiments also learn the input coefficients controlling the quantum rotation angles. We simulate the circuits locally and check selected calculations independently with Qiskit.

This question grew out of failures as well as improvements. A method that raises accuracy can still produce worse probability estimates. Updating a correction can help, waste measurements, or make a model overconfident.

## Findings so far

The numbers below describe specific simulated experiments. They are not guarantees for other tasks or hardware.

| Investigation | Observation | What it means |
| --- | --- | --- |
| Learned input encoding | On a diagonal-wave toy task, mean ideal accuracy rose from about 78.6% to 97.2%. | A useful task-specific representation can matter substantially. Other tasks gained little or regressed. [Checkpoint 40](quantum-first-run/research/quantum_nn/logs/40_checkpoint.json) |
| Noise-aware prediction-layer training | Five new training seeds on a ring task: accuracy 84.4% → 87.6%; confidently wrong predictions 8.4% → 2.3% at a 256-shot inference budget. | Training for finite measurement noise can help this task. Simple classical confidence controls explain part of the gain. [Checkpoint 49](quantum-first-run/research/quantum_nn/logs/49_checkpoint.json) |
| Grouping one fixed measurement record | Several noisy training views sometimes outperformed one averaged view at the same total training-shot budget. | How measurements become training data matters; grouping was not best across all tasks. This is an existing research idea, not a novelty claim. [Checkpoint 53](quantum-first-run/research/quantum_nn/logs/53_checkpoint.json) |
| Changing simulated device errors | On a parabola task after a channel change: 92.8% accuracy without device-specific correction, 86.2% with a stale correction, 95.5% after refreshing it. | An outdated correction can be harmful. These were hypothetical asymmetric readout/depolarization channels. [Checkpoint 53](quantum-first-run/research/quantum_nn/logs/53_checkpoint.json) |
| Reference-circuit monitoring | A 512-shot check detected a full simulated channel change in 99.85% of stationary test events, with 0.95% false alarms; smaller changes were frequently missed. | Monitoring can detect large changes in this setup. These are event rates for one baseline, not device-wide guarantees. [Experiment 54](quantum-first-run/research/quantum_nn/logs/54_result.json) |
| Triggered versus scheduled updates | On randomized timelines, triggered updates reached about 91.15% accuracy using 254,976 total shots; updates every six batches reached 90.94% using 262,144. The triggered method had worse cross-entropy. | There is an accuracy/cost tradeoff, not universal superiority. Monitoring also costs measurements when nothing changes. [Experiment 56](quantum-first-run/research/quantum_nn/logs/56_result.json) |
| Extreme confidence after recalibration | On fresh timelines, capping extreme logits reduced triggered-policy cross-entropy from 0.2360 to 0.2199 without changing accuracy or measurement cost; Brier error worsened slightly. | One safeguard can improve one metric while hurting another. A cap is not proof of calibrated probabilities. [Experiment 57](quantum-first-run/research/quantum_nn/logs/57_result.json) |

## Continued after publication

Experiments 59–60 were run after the initial GitHub push. They compare regularized affine recalibration, ordinary maximum-likelihood recalibration, and fixed confidence capping using calibration labels separate from encoder selection and test data.

In a paired calibration-budget test, ordinary recalibration's mean cross-entropy fell from **0.279 with 64 calibration examples to 0.197 with 256**, with the same test measurements. Calibration measurement cost increased from 16,384 to 65,536 shots per model/cohort. A fixed confidence cap helped cross-entropy at the smaller budget but hurt at 256 examples; Brier error did not improve with the cap. The 128-example cap result also differed across the two experiment batches, so we retain the uncertainty rather than declaring a universal rule.

Regularization selected using held-out calibration examples was not consistently better. These static-channel results still need temporal confirmation and fresh encoder-training seeds. See [experiment 60](quantum-first-run/research/quantum_nn/logs/60_result.json) and [checkpoint 61](quantum-first-run/research/quantum_nn/logs/61_checkpoint.json).

## What we have not established

- No uniquely quantum advantage, breakthrough, or improvement over the strongest possible classical methods has been demonstrated. The current two-qubit feature map has an exact elementary classical representation.
- The current monitoring studies reuse a small set of trained encoders on synthetic two-dimensional tasks. More measurement repeats do not create more independent training seeds or devices.
- Most encoder training uses ideal simulated expectations. Later tests model finite measurements and selected hypothetical errors; they do not reproduce a calibrated IBM backend.
- Earlier quantum-attention experiments are exploratory. We have not trained a practical quantum transformer or demonstrated broad reasoning, language-model capability, energy savings, or large-scale speedups.
- No hardware jobs were submitted for this research continuation. Original hardware-capable demos remain separate and require explicit real-job confirmation.
- Learning from logs means improving experimental decisions and preserving evidence. It does not mean retraining the underlying AI assistant.

## Research method

Before each experiment, review the preceding results and limitations. Record the hypothesis and protocol before running it. Keep training, selection/calibration, and test data separate where the protocol requires it, and disclose earlier experiments that reused selection data. Preserve failures, optimizer outcomes, random seeds, costs, and negative results.

Compare quantum components with classical equivalents and simple controls. Count all measurement shots, including feature views, basis settings, reference checks, and recalibration. Report accuracy together with probability errors such as cross-entropy and Brier score. Treat measurement repeats as dependent evaluations of the same trained models.

The JSON logs retain the original exploratory sequence, including imperfect protocols. Later corrections qualify earlier results rather than rewriting historical evidence. Source/result hash manifests help identify the exact versions used; earlier manifests can legitimately differ from files subsequently extended.

## Repository map

- [Quantum learning experiments and logs](quantum-first-run/research/quantum_nn/): QNNs, attention prototypes, feature/readout controls, finite-shot measurements, calibration, and monitoring.
- [Latest research checkpoint](quantum-first-run/research/quantum_nn/logs/61_checkpoint.json): findings, caveats, verification, and next actions.
- [Research handoff](quantum-first-run/research/HANDOFF.md): cumulative continuation notes.
- [Earlier research overview](quantum-first-run/research/README.md): original audit, circuit/spin studies, literature, and roadmap. Some documents describe earlier project stages; the latest checkpoint is the continuation guide.
- [Local circuit demonstrations](quantum-first-run/README.md): bit-frequency, Bell/GHZ population, and mirror diagnostics. Their outputs are not evidence of certified randomness or measured entanglement.
- [Earlier classical memory experiments](quantum-first-run/research/memory_lab/): preserved side investigations, not evidence of quantum advantage.
- `quantum-first-run/results/` and `research/results/`: historical local simulation outputs.

Virtual environments, caches, credentials, and local assistant state are excluded from version control. Research code, plans, results, historical artifacts, and provenance are included.

## Run locally

From the repository root, create an environment and install the recorded research dependencies:

```sh
python3 -m venv quantum-first-run/.venv
quantum-first-run/.venv/bin/python -m pip install -r quantum-first-run/research/requirements-research.txt
```

The dependency versions are pinned to the recorded development environment; a compatible Python/package platform is required. Package installation and the full experiment sequence have not been reverified on every platform.

Run independent gradient and circuit checks:

```sh
quantum-first-run/.venv/bin/python quantum-first-run/research/quantum_nn/validate_learned_inputs.py
```

Replay the exploratory sequence into a **new, empty log directory**:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
quantum-first-run/.venv/bin/python quantum-first-run/research/quantum_nn/continue.py \
  --logs /tmp/quantum-research-replay
```

A complete replay runs many sequential optimization and simulation studies. Individual continuation scripts depend on earlier result files and deliberately refuse to overwrite existing logs. The replay reproduces the experiment runners; editorial checkpoints and publication manifests are maintained separately.

## Next research steps

1. Consolidate the monitoring/calibration benchmark with fresh encoder-training seeds, independent calibration labels, and stronger fixed-schedule controls.
2. Test regularized or held-out calibration against confidence capping. Vary the label and shot budget instead of relying on one small calibration set.
3. Challenge the reference monitor with unseen, circuit-dependent and coherent errors, small changes, false-alarm accumulation, and realistic update delays.
4. Measure complete deployment costs. Avoid counting monitoring or calibration as free.
5. Read the closest related work in full and ask faculty whether a controlled replication or extension is worth pursuing.
6. Prepare a bounded IBM hardware experiment only when it answers a specific question that simulation cannot settle, with an explicit shot budget and classical controls.

## Related work

These methods build on existing ideas. The following are starting points, not a completed novelty review:

- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): temperature scaling and classical calibration controls.
- [Reduction of finite sampling noise in quantum neural networks](https://arxiv.org/abs/2306.01639): variance regularization for finite quantum measurements.
- [From Quantum Shots to Training Data](https://arxiv.org/abs/2604.28160): fixed-budget shot grouping and readout regularization. Our classification experiments differ from that paper's reservoir/forecasting protocol.
- [Generalization Error in Quantum Machine Learning in the Presence of Sampling Noise](https://arxiv.org/abs/2410.14654): learning with finite datasets and sampling noise.
- [Quantum Conformal Prediction](https://arxiv.org/abs/2304.03398): existing work on quantum-model uncertainty; our monitoring results do not establish its coverage guarantees.

This repository is a starting record for further testing and collaboration, not a finished scientific claim.
