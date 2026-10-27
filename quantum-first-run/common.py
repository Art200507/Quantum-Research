"""Shared helpers for backend selection, transpilation, job submission, and result I/O.

Centralizing this here means every experiment module submits circuits the same
way, which is what lets run.py batch circuits from all three experiments into a
single Sampler job. Batching does not guarantee a lower QPU usage charge.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.executor_sampler import Sampler as RuntimeSampler

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


@dataclass
class Submission:
    """One named circuit queued for execution, plus the slot it lands in."""

    label: str
    circuit: QuantumCircuit
    meta: dict[str, Any] = field(default_factory=dict)


def get_execution_backend(real: bool, min_num_qubits: int = 1):
    """Return (service_or_None, backend) that experiments execute against.

    Dry-run uses FakeTorino, a noise-calibrated classical model of an IBM
    Heron device — good enough to rehearse the exact submission code path
    (transpilation, Sampler, result shape) without spending any QPU time.
    Real mode asks the service for the least busy operational QPU.
    """
    if not real:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        return None, FakeTorino()

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=min_num_qubits
    )
    return service, backend


def get_ideal_sampler(seed: int | None = None) -> AerSampler:
    """A noiseless statevector Sampler, used as the ground-truth reference.

    Only ever invoked on circuits small enough to statevector-simulate
    (see mirror.py for why that cap is ~25-30 qubits, not 50+).
    """
    return AerSampler(seed=seed)


def transpile_for_backend(
    circuits: list[QuantumCircuit], backend, optimization_level: int = 3
) -> list[QuantumCircuit]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    return [pm.run(c) for c in circuits]


def run_batch(backend, circuits: list[QuantumCircuit], shots: int):
    """Submit every circuit in `circuits` as pubs of ONE Sampler job."""
    sampler = RuntimeSampler(mode=backend)
    job = sampler.run(circuits, shots=shots)
    return job


def get_counts_list(result) -> list[dict[str, int]]:
    """Extract per-circuit counts dicts from a Sampler result.

    Assumes every circuit was built with `.measure_all()`, which always
    names its classical register "meas".
    """
    return [pub_result.data.meas.get_counts() for pub_result in result]


def confirm_real_submission(backend_name: str, circuits: list[QuantumCircuit], shots: int) -> None:
    print("\n=== REAL HARDWARE SUBMISSION ===")
    print(f"Backend:            {backend_name}")
    print(f"Number of circuits: {len(circuits)}")
    print(f"Shots per circuit:  {shots}")
    print("Transpiled depths:")
    for i, c in enumerate(circuits):
        print(f"  circuit[{i}] ({c.name}): depth={c.depth()}, ops={sum(c.count_ops().values())}")
    print(
        "This consumes real QPU time. Verify your current plan, remaining usage, and job estimate before confirming."
    )
    answer = input('Type "yes" to submit this job, anything else to abort: ')
    if answer != "yes":
        raise SystemExit("Aborted: user did not confirm real-hardware submission.")


def save_run(payload: dict[str, Any], out_dir: str | None = None) -> str:
    directory = out_dir or ensure_results_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{timestamp()}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
