# Literature matrix and landscape

Search date: 25 October 2026. Primary publications, author manuscripts and official
docs only are used as evidence. Full text accessed means retrieved searchable
HTML, with relevant sections inspected; it does not mean every proof or supplement
was independently verified. Abstract-only entries cannot justify detailed method
claims. This is a bounded initial search, not a systematic review or novelty proof.

| ID and verified reference | Status/access | Method and reported finding | Limitations and relevance | Code/data availability |
|---|---|---|---|---|
| K23: Kim et al., [Evidence for the utility of quantum computing before fault tolerance](https://doi.org/10.1038/s41586-023-06096-3), Nature 618, 500–505 (2023). [Accessible full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10266970/) | Peer reviewed; full text via PMC after Nature access failed | 127-qubit heavy-hex Ising circuits; learned/twirled noise amplification and ZNE; checked tractable observables and compared MPS/isoTNS | Large circuit size is not advantage against all classical methods; circuit ideal is distinct from continuous Hamiltonian dynamics | [Figure data](https://figshare.com/articles/dataset/Evidence-for-the-utility-of-quantum-computing-before-fault-tolerance/22500355); end-to-end code not verified |
| BC23: Begušić and Chan, [Fast classical simulation of evidence for the utility of quantum computing before fault tolerance](https://arxiv.org/abs/2306.16372) (2023) | Preprint; abstract only | Sparse Pauli dynamics; reports laptop simulations agreeing with experimental estimates at much lower wall time | Early response; use converged successor for stronger evidence | Not separately verified |
| T24: Tindall, Fishman, Stoudenmire and Sels, [Efficient Tensor Network Simulation of IBM's Eagle Kicked Ising Experiment](https://arxiv.org/html/2306.14887v3), [PRX Quantum 5, 010308](https://doi.org/10.1103/PRXQuantum.5.010308) (2024) | Peer reviewed; author full text | Geometry-aware tensor networks with belief propagation; explains effectiveness using approximately tree-like correlations; extends dynamics | Approximate contraction/convergence must be assessed; favorable structure matters | Paper identifies [ITensorNetworks.jl](https://github.com/mtfishman/ITensorNetworks.jl) and [data](https://github.com/JoeyT1994/BP-TNS-Data); not executed here |
| BGC24: Begušić, Gray and Chan, [Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance](https://arxiv.org/html/2308.05077v3), [Science Advances 10, eadk4321](https://doi.org/10.1126/sciadv.adk4321) (2024) | Peer reviewed; author full text | SPD and mixed Schrödinger/Heisenberg tensor networks; converged studied observables more precisely than experimental estimates | Does not imply efficient simulation of arbitrary circuits, all states or all observables | Paper links [SPD](https://github.com/tbegusic/spd), [TN code](https://github.com/jcmgray/mixpic-bp-quantum-dynamics), and [data](https://github.com/tbegusic/arxiv-2308.05077-data); not run |
| E19: Endo, Zhao, Li, Benjamin and Yuan, [Mitigating algorithmic errors in Hamiltonian simulation](https://arxiv.org/abs/1808.03623), [PRA 99, 012334](https://doi.org/10.1103/PhysRevA.99.012334) (2019) | Peer reviewed; abstract and publisher search excerpt only | Noisy optimal Trotter-step trade-off and extrapolation of algorithmic errors, demonstrated numerically | Directly defeats novelty of merely observing an optimum | Code availability not verified |
| C23: Cirstoiu, Dilkes, Mills, Sivarajah and Duncan, [Volumetric Benchmarking of Error Mitigation with Qermit](https://quantum-journal.org/papers/q-2023-07-13-1059/), Quantum 7, 1059 (2023); [full text](https://arxiv.org/html/2204.09725v3) | Peer reviewed; full text | Hardware and simulated ZNE/CDR benchmarking; variability and overly optimistic mirror benchmarks | Mirror-to-target transfer is already an established research subject; simple depolarization can mispredict hardware | [Qermit](https://github.com/CQCL/qermit); paper links full circuit/results dataset |
| S26: Vicenzo Scavino Alfaro, [A finite-shot mean-squared-error boundary for Richardson zero-noise extrapolation](https://arxiv.org/html/2605.08251v2), arXiv:2605.08251v2 (15 Oct 2026) | Preprint; full text relevant sections | Fixed-shot bias–variance boundaries; observable-dependent behavior; reports Aer and IBM tests | Not peer reviewed as verified here; theorems/data not independently checked. Direct overlap with proposed fixed-budget question | Code/data archive declared in availability section; archive itself not tested |
| Z26: Zhou et al., [Joint Mitigation of Algorithmic and Physical Errors in Noisy Hamiltonian Simulation](https://arxiv.org/html/2609.11508v1) (10 Sep 2026) | Preprint; full text relevant sections | Coupled Trotter/noise extrapolation; reports superconducting Ising experiments and 100-qubit SPD | Assumptions include controllable noise and intrinsic floor; not independently reproduced. Defeats broad joint-mitigation novelty | Code archive not verified |
| M26: Maxwell et al., [Practical Estimation of Trotter Error for Hamiltonian Simulation](https://arxiv.org/abs/2606.30738) (2026) | Preprint; search abstract only | Reports improved practical Trotter-error estimation and software applications | Larger-scale/theoretical route exceeds starting mathematics; investigate before claiming new resource estimators | Software claimed; specific implementation not verified |
| IBM26: [Plans overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview), [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2), [Runtime local testing](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/dev/runtime-service) | Official vendor documentation, accessed live | Plan allowance, primitive interfaces, fake-backend local mode | Documents change; dev docs differ from pinned installed API; no account-specific allowance inferred | Local installed source and smoke run also inspected |

## What the 2023 experiment actually means

K23 uses H=-J sum_<ij> ZiZj+h sum_i Xi from all-zero spins on heavy hex.
Its fixed RZZ angle is -pi/2; variable RX angle controls the kicks. It treats
the digital circuit as the reference, explicitly setting aside Trotter model
error. Measured quantities include magnetization and selected weight-10/17 Pauli
strings. Learned Pauli-Lindblad noise, twirling and probabilistic amplification
support extrapolation at gains 1, 1.2 and 1.6. Exact/Clifford checks and selected
classical approximations support the accuracy claim, not a proven practical
speedup. [K23 full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10266970/)

BC23, T24 and BGC24 challenged the classical baseline rather than denying the
hardware ran. They exploit operator sparsity and geometry rather than storing
a generic full wavefunction. BGC24 converges the tested observables and identifies
experimental extrapolation inaccuracies. Later Z26 reports joint mitigation on
hardware, while S26 studies finite-shot decisions. Neither is an independent
exact reproduction of K23; both are relevant follow-up methods and preprints.
[Cited response](https://arxiv.org/html/2308.05077v3),
[joint mitigation](https://arxiv.org/html/2609.11508v1),
[finite-shot study](https://arxiv.org/html/2605.08251v2).

Our chain shares noncommuting ZZ and X terms, but has 3–4 spins, open 1D geometry,
positive ZZ coupling, small continuous target times, and deliberately controlled
product-formula steps. It is not a miniature reproduction of IBM's utility claim.
Here exact exponentiation is a cheap classical ground truth, and the pure
transverse-field chain additionally has integrable structure. Increasing its
size is not a credible advantage project.

## Concepts needed for choosing a question

An observable is a quantity such as the average Z spin, not the whole state.
A Pauli measurement yields +1 or -1; its average estimates an expectation value.
Trotterization alternates evolutions of simpler pieces of H, because ZZ and X
cannot generally be combined by just adding their separate gate effects.
More steps reduce this approximation error while adding noisy operations.
Mitigation combines noisy averages to estimate a less biased observable; it
does not repair each shot or prepare a higher-fidelity state. Extrapolation's
negative weights amplify sampling uncertainty. The correct metric for this
pilot is MSE=bias squared+variance, measured against a declared target.

Searches combined the 2023 paper title with classical simulation, Trotter/noise
optimal depth, fixed-shot ZNE, mirror application prediction, and 2025/2026
qualifiers. New work was found that directly overlaps the most tempting ideas.
Layout-specific transfer, resource cost including calibration, and held-out
robustness remain possibilities to investigate, not verified gaps. Before any
paper plan, inspect the full E19/S26/Z26 methods and code, forward citations,
and related 2025–2026 application-oriented benchmarking papers. The search
cannot establish that no redundant paper exists.
