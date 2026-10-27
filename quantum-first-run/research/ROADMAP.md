# Two-week plan and supervised roadmap

Background reported: intermediate Python; no linear algebra, probability or quantum
coursework; University at Buffalo. Time availability not given. The plan assumes
6–8 hours/week and should be extended if fewer hours are available. Becoming
comfortable with the mathematics may take longer than two weeks. A paper in
8–12 weeks is not promised.

## First two weeks

1. Days 1–3 (3 hours): complex numbers, vectors, matrix multiplication and normalized
   probabilities. Calculate H|0>, X|0>, Z|0> and two-qubit tensor products by hand,
   then verify using NumPy. Explain Qiskit's qubit/string ordering.
2. Days 4–5 (2 hours): expectation, variance, binomial sampling and standard error.
   Derive Var(mean of ±1 shots)=(1-mu²)/N and the variance of a weighted sum.
3. Days 6–7 (2 hours): understand exact evolution versus product-formula evolution.
   Reproduce the zero-time and depolarizing checks; inspect 1- and 2-step circuits.
4. Days 8–10 (3 hours): run pilot.py and analyze.py. Reconstruct the 8-step example
   in RESULTS.md from CSV without copying its prose. Explain why a lower bias can
   give higher MSE, and why signed approximation/noise biases can cancel.
5. Days 11–12 (2 hours): read K23's experiment/reference distinction, C23's mirror
   discussion, and S26's estimator setup. Make a one-page “what is already known”
   note, marking proofs you cannot yet follow.
6. Days 13–14 (2 hours): prepare a five-minute explanation with one figure, an
   honest background statement and a request for supervised replication or a
   better scoped question. Use CONCEPT_NOTE.md; do not present generated code as
   mathematics you independently understand yet.

Exit criterion: explain the Hamiltonian, two gate angles, chosen estimator, one
negative result and two limitations without relying on assistant wording. If this
is not comfortable, spend another two weeks on foundations before extending code.

## Eight-to-twelve-week supervised path

| Weeks | Work | Decision/evidence |
|---|---|---|
| 1–2 | Foundations and independent reproduction | One trusted figure and hand derivation |
| 3–4 | Read closest code/methods with supervisor; replace parameter scaling by preserved folding and compile to native gates | Drop novelty claims already covered; verify amplified circuits preserve ideal observable |
| 5–6 | Add damping/coherent noise, readout and field/initial-state sensitivity; predeclare held-out settings | Does the effect survive realistic model changes? |
| 7–8 | Budgeted training/selection with calibration overhead; raw, oracle and practical baselines | Held-out MSE comparison with repeats/intervals and honest cost ledger |
| 9–10 | Replicate failures, robustness and external code comparison; optional small hardware proposal only if justified | Supervisor decides whether there is a contribution or only replication |
| 11–12 | Clean reproducibility package and short research report; decide next scientific question | A negative result or a pivot can be the correct endpoint |

## University at Buffalo fit

[Jamir Marino's official UB profile](https://www.buffalo.edu/innovation/quantum/about-us/faculty.host.html/content/shared/arts-sciences/physics/faculty-profiles-directories/new-profiles/marino-jamir.html)
lists nonequilibrium many-body physics, driven-open systems and quantum simulators.
That makes him a plausible person to ask whether a small spin-dynamics replication
could lead to a scientifically useful question; this is an inference about fit,
not knowledge of openings.

[Xuedong Hu's official profile](https://arts-sciences.buffalo.edu/physics/about/leadership-administration.host.html/content/shared/arts-sciences/physics/faculty-profiles-directories/new-profiles/hu-xuedong.html)
focuses on quantum coherence and information processing in semiconductor devices.
He is a plausible alternative for noise/control guidance or referral, though an
IBM gate-model mitigation project may not match his immediate research priorities.
The [undergraduate research mentor page](https://arts-sciences.buffalo.edu/physics/undergraduate/research-mentors.html)
also identifies relevant theoretical mentors. Check current interests and ask for
advice about prerequisites. Nobody was contacted and availability was not verified.
