# Research starting point — 25 October 2026

**Recommendation:** start with a resource-aware spin-dynamics replication asking
when a better raw circuit beats error mitigation. No established novelty, quantum
advantage or hardware result is claimed.

Read in this order:

1. [Audit](AUDIT.md): what the existing code measures and corrections made.
2. [Actual results](RESULTS.md) and [figure](results/mse.png).
3. [Literature matrix and verified references](LITERATURE.md), [BibTeX](references.bib).
4. [Five candidate questions and novelty challenges](CANDIDATES.md).
5. [Recommended project](PROJECT.md), [learning plan and UB fit](ROADMAP.md).
6. [Professor-facing concept note](CONCEPT_NOTE.md).
7. [Handoff and exact commands](HANDOFF.md).

From `/Users/AtharvaBadgujar/Quantum/quantum-first-run`:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantum-mpl .venv/bin/python research/pilot.py
.venv/bin/python research/analyze.py
.venv/bin/python research/checks/audit_local.py
```

Use the existing environment. `requirements-research.txt` records the direct
scientific pins used, not a full transitive lock. Python/platform versions and
seed are in `results/metadata.json`. Numerical files are overwritten on rerun;
copy them to a named run directory first if preserving another run. No command
above initializes an IBM service, reads credentials, or submits remote jobs.

The pilot generated 4320 estimator rows, exact references, simulated noise means,
analytic MSE, 2000 synthetic trials per row and uncertainty on sampled MSE.
`analyze.py` also computes resource sensitivity using an explicitly abstract
gate-shot budget. Noise parameter scaling is an oracle baseline, so these
results do not estimate real QPU cost or validate practical noise amplification.

**What to investigate first, and why:** understand and independently reproduce
the fixed-budget comparison. It teaches the measurements and error accounting
needed for credible experiments, with informative negative outcomes on a laptop.
**What could invalidate an extension:** existing S26/Z26 coverage, loss of gain
against tuned raw baselines, realistic amplification costs or noise mismatch.
**What makes it professor-assessable:** one explained and independently rerun
figure, a close-prior-work comparison, explicit limitations and a bounded held-out
evaluation protocol. A supervisor can then decide whether to extend or pivot.
