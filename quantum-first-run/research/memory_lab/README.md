# Memory lab: ten completed adaptive experiments

Start with [REPORT.md](REPORT.md) for the results in easy words, then
[JOURNAL.md](JOURNAL.md) for each result-driven decision and
[SOURCES.md](SOURCES.md) for prior work and access limitations.

The central result is a useful **classical baseline**, not a new quantum neural
network: a quantized trained linear layer recalls 256 random associations in the
tested mild conditions using 4,100 bytes of packed weights. It loses robustness
as load/query corruption increases. Phases, interference-style binding and qubit
encoding did not show an advantage in these tests.

## Files

- lab.py: experiments 1–7, core models, repeated memories, algebra checks.
- extended.py: experiment 8, replacement/update behavior and rebuild check.
- quantum_check.py: experiment 9, 672 three-qubit circuits and readout comparison.
- validate.py: experiment 10, frozen methods on fresh seeds and timing diagnostics.
- results/01.json … 10.json: actual rows, seed summaries, settings and costs.
- results/heldout.png: standalone figure of capacity/noise limits.
- results/manifest.json: dependencies, Python/platform, source/result hashes.
- reproduce.py: sequential replay into an empty directory; it never submits jobs.

## Reproduce

From `/Users/AtharvaBadgujar/Quantum/quantum-first-run` with the existing environment:

```sh
.venv/bin/python research/memory_lab/reproduce.py --out /tmp/memory-lab-replay
MPLCONFIGDIR=/tmp/quantum-mpl .venv/bin/python research/memory_lab/make_report.py
```

Use a new or empty replay directory. Original numerical results are protected
against overwrite. The figure script reads the committed-in-workspace results;
there is no Git repository here. Numerical replays should agree to floating-point
tolerance; timings vary. Direct dependency versions are in the manifest and the
parent research/requirements-research.txt. No new package installation was needed.

## Handoff / next action

All ten authorized experiments are complete. Scope deliberately stayed small;
no unrelated infrastructure, paid compute, credential access or hardware jobs.
Existing demonstrations, hardware safeguards and earlier research artifacts were
not modified by this sprint. A source/result manifest and this folder preserve
continuity. The earlier spin-dynamics investigation remains a separate direction.

Next useful research task: reproduce a published structured associative-recall
benchmark and evaluate absent/ambiguous keys plus mixed updates/queries, including
encoder costs. Compare modern linear/attention memories and quantized baselines.
Keep a real/imaginary ablation; do not presume a quantum ingredient is necessary.
No publishable gap has been established. Do not proceed to hardware or full-scale
LLM training based on these tiny experiments.
