"""Mirror-circuit return-probability diagnostic, not a classical-hardness test.

U followed by its inverse has a known all-zero ideal output. Generic rotations
are non-Clifford, but this fixed-depth 1D circuit admits tensor-network methods;
statevector memory estimates do not bound all classical algorithms. A barrier
preserves separation of the two halves for the tested compiler, not every gate.
Noisy counts combine gate, preparation and measurement effects and may miss
coherent errors that cancel in the inverse. Local caps are conservative policy.
"""
from __future__ import annotations

import math
import random

from qiskit import QuantumCircuit

from common import Submission

# Largest N for the noiseless Aer correctness check (confirms U;U^-1 == I).
# 2**25 amplitudes (~0.5 GB) is comfortable on a laptop; 2**30+ starts to hurt.
STATEVECTOR_CAP = 25

# Largest N we'll actually execute against the noisy FakeTorino model in
# dry-run. Much lower than STATEVECTOR_CAP -- see the module docstring for
# measured timings showing why Monte Carlo noise trajectories, not memory,
# are the binding constraint here.
NOISY_LOCAL_CAP = 10

# Fixed logical depth; routing and noise costs still depend on size and layout.
LAYERS = 8


def _build_mirror_unitary(n_qubits: int, layers: int, rng: random.Random) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        for q in range(n_qubits):
            gate = rng.choice(["sx", "rz", "ry", "rx"])
            if gate == "sx":
                qc.sx(q)
            else:
                theta = rng.uniform(0, 2 * math.pi)  # non-Clifford for rz/ry/rx
                getattr(qc, gate)(theta, q)
        offset = layer % 2  # brick-wall pattern: alternate pairing each layer
        for q in range(offset, n_qubits - 1, 2):
            if rng.random() < 0.5:
                qc.cx(q, q + 1)
            else:
                qc.ecr(q, q + 1)
    return qc


def _build_mirror_circuit(n_qubits: int, layers: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    u = _build_mirror_unitary(n_qubits, layers, rng)
    # Without a barrier here, the transpiler's gate-cancellation passes notice
    # that U is immediately followed by U^-1 -- an exact palindrome of mutual
    # inverse pairs -- and recursively cancel the ENTIRE circuit down to
    # nothing from the middle out (verified empirically: optimization_level=3
    # collapsed a 100-qubit mirror circuit to depth 1). The barrier blocks
    # commutation/cancellation across the U / U^-1 boundary while still
    # allowing full SABRE-optimized layout and routing on each half.
    full = u.copy()
    full.barrier()
    full.compose(u.inverse(), inplace=True)
    full.measure_all()
    full.name = f"mirror_n{n_qubits}"
    return full


def build(sizes: tuple[int, ...] = (5, 20, 50, 100), layers: int = LAYERS, seed: int = 1234) -> list[Submission]:
    subs = []
    for n in sizes:
        circuit = _build_mirror_circuit(n, layers, seed=seed + n)
        subs.append(
            Submission(
                label=f"mirror_n{n}",
                circuit=circuit,
                meta={
                    "n_qubits": n,
                    "simulate_locally": n <= STATEVECTOR_CAP,  # noiseless ideal check
                    "noisy_locally": n <= NOISY_LOCAL_CAP,  # actual FakeTorino execution
                },
            )
        )
    return subs


def analyze(
    counts_list: list[dict[str, int] | None],
    submissions: list[Submission],
    ideal_counts_map: dict[str, dict[str, int]],
    out_dir: str,
) -> dict:
    results = {}
    ns, fidelities, hammings = [], [], []
    skipped_ns = []

    print("\n--- Mirror circuit return diagnostic ---")
    for sub, counts in zip(submissions, counts_list):
        n = sub.meta["n_qubits"]
        if counts is None:
            skipped_ns.append(n)
            ideal = ideal_counts_map.get(sub.label)
            suffix = ""
            if ideal:
                ideal_shots = sum(ideal.values())
                ideal_frac = ideal.get("0" * n, 0) / ideal_shots
                suffix = f"  [ideal/noiseless check: {ideal_frac:.3f}]"
            print(
                f"N={n}: skipped locally (too costly to simulate noisily, see mirror.py) "
                f"-- omitted by local simulation policy{suffix}"
            )
            results[sub.label] = {"n_qubits": n, "skipped": True, "ideal_all_zero_fraction": ideal_frac if ideal else None}
            continue

        shots = sum(counts.values())
        zero_bits = "0" * n
        all_zero_frac = counts.get(zero_bits, 0) / shots
        avg_hamming = sum(bs.count("1") * c for bs, c in counts.items()) / shots

        ideal = ideal_counts_map.get(sub.label)
        ideal_frac = None
        if ideal:
            ideal_shots = sum(ideal.values())
            ideal_frac = ideal.get(zero_bits, 0) / ideal_shots

        suffix = f"  [ideal/noiseless check: {ideal_frac:.3f}]" if ideal_frac is not None else ""
        print(f"N={n}: all-zero fraction={all_zero_frac:.4f}  avg Hamming weight={avg_hamming:.3f}{suffix}")

        ns.append(n)
        fidelities.append(all_zero_frac)
        hammings.append(avg_hamming)
        results[sub.label] = {
            "n_qubits": n,
            "skipped": False,
            "all_zero_fraction": all_zero_frac,
            "avg_hamming_weight": avg_hamming,
            "ideal_all_zero_fraction": ideal_frac,
        }

    _plot(ns, fidelities, skipped_ns, out_dir)
    return results


def _plot(ns: list[int], fidelities: list[float], skipped_ns: list[int], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if ns:
        ax.plot(ns, fidelities, "o-", label="all-zero fraction (measured)")
    for n in skipped_ns:
        ax.axvline(n, color="gray", linestyle=":", alpha=0.6)
        ax.text(n, 0.02, f"N={n}\nnot locally\nsimulated", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("N (qubits)")
    ax.set_ylabel("observed all-zero return probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Mirror return probability vs N")
    if ns:
        ax.legend()

    fig.tight_layout()
    path = f"{out_dir}/mirror_fidelity_vs_n.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved plot: {path}")
