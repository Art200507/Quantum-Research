# Prospective undergraduate replication: resource-aware spin-dynamics estimation

I am a University at Buffalo undergraduate with intermediate Python experience,
beginning the linear algebra, probability and quantum mechanics needed for
research. I would like supervision on a focused replication and help determining
whether a worthwhile extension exists.

**Question.** When does zero-noise extrapolation improve a spin observable's
mean-squared error at a fixed resource budget, compared with choosing a better
unmitigated Trotter circuit?

**Prior work.** Endo et al., PRA 99, 012334 (2019) already studied competing
algorithmic/physical errors. Cirstoiu et al., Quantum 7, 1059 (2023) benchmarked
mitigation and its variability. Recent preprints arXiv:2605.08251v2 and
arXiv:2609.11508 address finite-shot and joint-error mitigation directly. I am
not claiming the question or the observed trade-off is new. Verified links and
access limitations are in LITERATURE.md.

**Method and preliminary evidence.** An assistant-developed local pilot compares
exact matrix evolution with first-/second-order circuits for 3–4 open-chain spins,
H=sum ZiZ(i+1)+0.7 sum Xi, initially all zero. It separates product-formula bias,
modeled depolarizing noise and shot variance. Two-scale linear extrapolation uses
the same total shots as raw estimation. For 4 spins, t=1, second order, 8 steps,
p=.005 per abstract RZZ and Z0, extrapolation with a 3:1 shot split reduces bias
magnitude from .01260 to .004071 but raises MSE from .004392 to .01707 at 200 shots.
At 20000 shots it slightly beats raw at that depth, yet loses to a shallower raw
circuit in the tested grid. Numerical checks and seeded Monte Carlo accompany
machine-readable data. I still need to independently understand and reproduce
these calculations.

**Limitations.** This is simulated evidence from tiny, classically tractable
systems. Noise gain is an oracle model parameter; there is no native-gate folding,
calibration overhead, drift or hardware validation. No advantage or novelty is
claimed. The current work is an initial methodological replication, not a full
reproduction of any one paper.

**Possible extension and supervision needed.** Subject to a closer redundancy
review, test whether a small, fully charged calibration budget can select
circuit depth and mitigation that generalize to held-out times and noise profiles.
I would value guidance on prerequisites, an appropriate physical model, relevant
existing code and whether this evaluation merits extension or a different project.
A rigorous negative result or a better-scoped replication would be welcome.
