# quantum-first-run

Three small experiments on IBM Quantum hardware, built to spend as little of
your Open Plan's **10 minutes of QPU time per 28 days** as possible:

- **qrng.py** — quantum random number generator (Hadamard + measure, 8 qubits).
- **mirror.py** — a "beyond classical" mirror-circuit demo at N = 5, 20, 50, 100 qubits.
- **bell_ghz.py** — Bell pair and 3-qubit GHZ state, the standard sanity check.

`run.py` builds circuits from whichever of those you select, batches **all of
them into a single Sampler job**, and only ever touches real hardware if you
pass `--real` and then type `yes` at a confirmation prompt that shows you the
backend, shot count, and transpiled circuit depths first.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then save your IBM Quantum Platform credentials once:

```bash
python setup_account.py
```

This prompts for your API key (hidden input, via `getpass`) and your instance
CRN, then calls `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", ...)`.
Credentials are written by Qiskit to `~/.qiskit/qiskit-ibm.json` — never
printed, and never stored in this project. Get your API key and CRN from
https://quantum.cloud.ibm.com/ under your instance's API key page.

## Running

**Dry-run (default, free, instant)** — runs every circuit on `FakeTorino`, a
noise-calibrated classical model of a real IBM Heron device, so you can check
the whole pipeline (circuit construction → transpile → Sampler → analysis →
plots) without spending any QPU time:

```bash
python run.py
```

**Real hardware** — add `--real`. You'll see the backend, number of circuits,
shots, and transpiled depths, and must type `yes` to proceed:

```bash
python run.py --real
```

Useful flags (all work in both modes):

```bash
python run.py --experiments qrng,bell_ghz   # skip the mirror demo
python run.py --mirror-sizes 5,20           # skip the expensive large-N mirror circuits
python run.py --shots 500                   # fewer shots per circuit
```

After a real run, the job ID and reported QPU usage (seconds) are printed and
saved. Raw counts, metadata, and derived stats for every run (dry or real) are
written to `results/<timestamp>.json`; plots go to `results/*.png`.

## What each experiment shows

**qrng.py** — Hadamard on 8 qubits puts each one in an equal superposition;
measuring collapses each shot to a uniformly random 8-bit string, i.e. one
random byte. 1000 shots → 1000 bytes. We run a per-bit frequency/chi-square
test (each bit position should show ~50/50 split) and an overall bitstream
test, with exact p-values via the closed-form chi-square(1 dof) survival
function (no scipy needed) — then run the identical test on Python's `random`
module for comparison.

**mirror.py** — builds a random circuit `U` (single-qubit rotations mixed with
CX/ECR entanglers), appends `U.inverse()`, and measures. Since `U @ U⁻¹ = I`
*exactly*, the noiseless answer is the all-zero bitstring with certainty —
which means we know the correct answer without ever needing to classically
simulate the circuit, and can therefore meaningfully probe hardware at sizes
where classical simulation is impossible. We report the fraction of all-zero
outcomes and average Hamming weight per N, and plot fidelity (all-zero
fraction) vs N.

In dry-run, local execution is capped well below the default sizes (5, 20,
50, 100), for two *different* reasons:
- **N = 5 and 20** are cheap to verify noiselessly on Aer (confirms the
  circuit construction is correct — fidelity should come back ~1.0), but only
  **N = 5** is also run through the *noisy* FakeTorino model: simulating with
  a realistic noise model means Aer has to re-run the whole circuit once per
  shot (a fresh random "trajectory" each time), and that cost wall arrives
  around N≈10-15 on a laptop — long before N=20 or the memory limit below.
- **N = 50 and 100** are never simulated locally at all, noisy or noiseless —
  a statevector simulator needs 2^N complex amplitudes regardless of what the
  circuit computes, and 2^50 already needs ~18 petabytes of RAM (2^100 is
  beyond absurd). They're still transpiled in dry-run (so you get a realistic
  depth/gate-count preview), just not executed.

In `--real` mode none of this applies — every requested size is sent to real
hardware, which is the whole point of the exercise.

**bell_ghz.py** — the two textbook entangled states, run on the backend and
cross-checked against the noiseless Aer simulator side by side. Any
population outside `{00,11}` (Bell) or `{000,111}` (GHZ) is readout/gate
error — this is the thing to check first on any new backend before trusting
anything else.

## Roughly how much runtime a real run uses

A real job's actual cost is only known for certain after it runs — `run.py`
prints and saves it (`job.usage()`, seconds) — but as a rough planning
budget for the **default** run (`qrng` + `bell` + `ghz` + mirror at
5/20/50/100, 1000 shots each, batched into one job):

| circuit            | shots | rough QPU time |
|---------------------|-------|----------------|
| qrng (8q, shallow)   | 1000  | a few seconds |
| bell (2q), ghz (3q)  | 1000 each | a few seconds total |
| mirror N=5, N=20     | 1000 each | a few seconds each |
| mirror N=50, N=100   | 1000 each | **the bulk of the cost** — thousands of gates post-transpile, likely 10-30s each |

So a full default run likely costs somewhere in the **30-90 second** range out
of your 600-second (10 min) budget — but this is a rough estimate, not a
guarantee. To be conservative on a first real run:

```bash
python run.py --real --mirror-sizes 5,20     # cheap: skip the expensive large-N circuits first
python run.py --real --experiments mirror --mirror-sizes 100 --shots 200   # then try N=100 alone, fewer shots
```

and check the printed/saved `usage_seconds` after each run to calibrate your
remaining budget before running larger batches.
