"""Bell pair and 3-qubit GHZ state: the standard sanity check for a new backend.

Ideal outcome for both: only the all-0 and all-1 bitstrings appear, each with
~50% probability. Any weight on other bitstrings on real/noisy hardware is
readout or gate error, which is why this is run against the noiseless Aer
simulator in the same breath for comparison.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from common import Submission


def build() -> list[Submission]:
    bell = QuantumCircuit(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    ghz = QuantumCircuit(3, name="ghz3")
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    ghz.measure_all()

    return [
        Submission(label="bell", circuit=bell, meta={"n_qubits": 2, "expected": {"00", "11"}}),
        Submission(label="ghz3", circuit=ghz, meta={"n_qubits": 3, "expected": {"000", "111"}}),
    ]


def _expected_fraction(counts: dict[str, int], expected: set[str]) -> float:
    total = sum(counts.values())
    in_support = sum(n for bs, n in counts.items() if bs in expected)
    return in_support / total if total else 0.0


def analyze(
    counts_list: list[dict[str, int]],
    submissions: list[Submission],
    ideal_counts_list: list[dict[str, int]],
    out_dir: str,
) -> dict:
    results = {}
    print("\n--- Bell / GHZ sanity check ---")
    for sub, counts, ideal_counts in zip(submissions, counts_list, ideal_counts_list):
        frac = _expected_fraction(counts, sub.meta["expected"])
        ideal_frac = _expected_fraction(ideal_counts, sub.meta["expected"])
        print(f"{sub.label}: backend counts={counts}")
        print(f"{sub.label}: ideal (Aer) counts={ideal_counts}")
        print(
            f"{sub.label}: fraction in {{{','.join(sorted(sub.meta['expected']))}}} "
            f"-> backend={frac:.3f}  ideal={ideal_frac:.3f}"
        )
        results[sub.label] = {
            "backend_counts": counts,
            "ideal_counts": ideal_counts,
            "backend_expected_fraction": frac,
            "ideal_expected_fraction": ideal_frac,
        }

    _plot(submissions, counts_list, ideal_counts_list, out_dir)
    return results


def _plot(submissions, counts_list, ideal_counts_list, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(submissions), figsize=(5.5 * len(submissions), 4))
    if len(submissions) == 1:
        axes = [axes]

    for ax, sub, counts, ideal_counts in zip(axes, submissions, counts_list, ideal_counts_list):
        labels = sorted(set(counts) | set(ideal_counts))
        backend_vals = [counts.get(l, 0) for l in labels]
        ideal_vals = [ideal_counts.get(l, 0) for l in labels]
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], backend_vals, width, label="backend")
        ax.bar([i + width / 2 for i in x], ideal_vals, width, label="ideal (Aer)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("counts")
        ax.set_title(sub.label)
        ax.legend()

    fig.tight_layout()
    path = f"{out_dir}/bell_ghz_counts.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved plot: {path}")
