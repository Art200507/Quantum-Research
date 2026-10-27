"""Quantum random number generator: H + measure on 8 qubits, 1000 shots -> random bytes.

Each shot of an 8-qubit all-Hadamard circuit collapses to one uniformly random
8-bit string, i.e. one random byte. We check those bytes for bias
with a per-bit chi-square test and compare against Python's PRNG (`random`
module) run through the identical test.
"""
from __future__ import annotations

import math
import random as pyrandom

from qiskit import QuantumCircuit

from common import Submission


def build(n_qubits: int = 8) -> list[Submission]:
    qc = QuantumCircuit(n_qubits, name="qrng")
    qc.h(range(n_qubits))
    qc.measure_all()
    return [Submission(label="qrng", circuit=qc, meta={"n_qubits": n_qubits})]


def _chi2_p_value_df1(chi2_stat: float) -> float:
    """P(X > chi2_stat) for X ~ chi-square(1 dof).

    chi2(1) is the square of a standard normal, so its survival function has a
    closed form in terms of erfc -- no need for a scipy dependency.
    """
    return math.erfc(math.sqrt(chi2_stat / 2))


def _bit_frequency_chi2(bits: list[int]) -> tuple[float, float]:
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    expected = n / 2
    chi2 = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
    return chi2, _chi2_p_value_df1(chi2)


def _analyze_bitstream(bitstrings: list[str], source_name: str) -> dict:
    n_qubits = len(bitstrings[0])
    shots = len(bitstrings)

    # Per-qubit-position frequency test: each position should be ~50/50.
    per_bit = []
    for pos in range(n_qubits):
        bits = [int(s[pos]) for s in bitstrings]
        chi2, p = _bit_frequency_chi2(bits)
        per_bit.append({"position": pos, "ones_fraction": sum(bits) / shots, "chi2": chi2, "p_value": p})

    # Whole-stream frequency test: flatten every bit from every shot.
    all_bits = [int(ch) for s in bitstrings for ch in s]
    stream_chi2, stream_p = _bit_frequency_chi2(all_bits)

    byte_values = [int(s, 2) for s in bitstrings]

    print(f"\n--- {source_name}: frequency / chi-square check ({shots} shots, {n_qubits} bits) ---")
    for row in per_bit:
        flag = "not rejected" if row["p_value"] > 0.01 / n_qubits else "reject (Bonferroni family alpha=0.01)"
        print(
            f"  bit {row['position']}: P(1)={row['ones_fraction']:.3f}  "
            f"chi2={row['chi2']:.3f}  p={row['p_value']:.3f}  [{flag}]"
        )
    print(f"  whole stream ({len(all_bits)} bits): chi2={stream_chi2:.3f}  p={stream_p:.3f}")

    return {
        "source": source_name,
        "shots": shots,
        "n_bits": n_qubits,
        "per_bit": per_bit,
        "stream_chi2": stream_chi2,
        "stream_p_value": stream_p,
        "byte_values": byte_values,
    }


def analyze(counts_list: list[dict[str, int]], submissions: list[Submission], out_dir: str) -> dict:
    counts = counts_list[0]
    n_qubits = submissions[0].meta["n_qubits"]

    # Expand counts for marginal frequency analysis; temporal shot order is lost.
    bitstrings = [bs for bs, n in counts.items() for _ in range(n)]
    shots = len(bitstrings)

    # Frequency tests do not certify unpredictability or quantum origin.
    quantum_result = _analyze_bitstream(bitstrings, "Quantum (QPU/simulator)")

    classical_bitstrings = [format(pyrandom.getrandbits(n_qubits), f"0{n_qubits}b") for _ in range(shots)]
    classical_result = _analyze_bitstream(classical_bitstrings, "Classical (Python random)")

    _plot(quantum_result, classical_result, out_dir)

    return {"quantum": quantum_result, "classical": classical_result}


def _plot(quantum_result: dict, classical_result: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    positions = [r["position"] for r in quantum_result["per_bit"]]
    q_frac = [r["ones_fraction"] for r in quantum_result["per_bit"]]
    c_frac = [r["ones_fraction"] for r in classical_result["per_bit"]]
    width = 0.35
    axes[0].bar([p - width / 2 for p in positions], q_frac, width, label="quantum")
    axes[0].bar([p + width / 2 for p in positions], c_frac, width, label="python random")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("bit position")
    axes[0].set_ylabel("P(bit = 1)")
    axes[0].set_title("Per-bit frequency (should be ~0.5)")
    axes[0].legend()

    axes[1].hist(quantum_result["byte_values"], bins=32, alpha=0.6, label="quantum")
    axes[1].hist(classical_result["byte_values"], bins=32, alpha=0.6, label="python random")
    axes[1].set_xlabel("byte value (0-255)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Byte value distribution")
    axes[1].legend()

    fig.tight_layout()
    path = f"{out_dir}/qrng_frequency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved plot: {path}")
