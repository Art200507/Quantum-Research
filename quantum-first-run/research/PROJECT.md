# Recommended project and provisional contribution

Investigate **when choosing a better raw circuit beats applying error mitigation
for a dynamical observable at a fixed resource budget**. Begin with the included
replication, not an original-method claim. The actual pilot finds bias–variance
trade-offs and shows why a same-depth comparison can overstate mitigation value.
See RESULTS.md for exact settings and numbers.

This fits current resources: intermediate Python, a working local Qiskit environment,
no prior university quantum/mathematics coursework reported, and no hardware data.
The first valuable contribution is a reproducible, understood baseline, with the
ideal digital circuit and continuous-H target clearly distinguished. This is enough
to ask a professor for project advice, but not enough to claim a publishable gap.

A possible supervised extension asks whether a small calibration/training budget
can select depth/order/mitigation on held-out evolution times and noise conditions
more reliably than simple raw baselines. The test must charge selection costs,
replace oracle noise scaling and compare against recent S26/Z26 code. Novelty is
unestablished; those papers could make the extension redundant.

A concrete professor-assessable package consists of one independently reproduced
figure, a derivation of MSE and the gate angles, known failure cases, a literature
overlap table and a bounded held-out protocol. A useful outcome can be “mitigation
is not beneficial in this regime.” If no worthwhile unresolved question remains,
accept this as replication and pivot with a supervisor. Fallback: candidate 2's
controlled compilation/layout comparison.

Hardware is not justified yet: oracle amplification, native compilation,
calibration costs and model mismatch are unresolved. No hardware plan or usage
estimate is presented as executable. Before any future proposal, require a robust
local gain over tuned raw baselines under at least a coherent and an incoherent
noise family, an explicit unresolved question approved by the supervisor, a native
compiled circuit inventory (depth, two-qubit count, layout and measurement map),
shot counts including calibration/twirling/repeats, and an uncertainty-aware usage
estimate using current backend timings and account allowance. Then define a tiny
first run and a stopping rule for loss of gain, drift or usage overrun. Existing
`--real` plus literal `yes` safeguards remain; no hardware submission is authorized
by this document.
