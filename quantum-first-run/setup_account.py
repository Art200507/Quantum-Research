#!/usr/bin/env python
"""One-time setup: save your IBM Quantum Platform credentials to disk so that
`QiskitRuntimeService()` can load them automatically in run.py.

This prompts for your API key (hidden input) and CRN, then calls
QiskitRuntimeService.save_account(...). Credentials are written by Qiskit to
~/.qiskit/qiskit-ibm.json -- NOT printed to the terminal and NOT stored
anywhere in this project.

Get your API key and CRN (Cloud Resource Name) from:
https://quantum.cloud.ibm.com/ -> your instance -> API key / CRN
"""
from __future__ import annotations

import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    print("IBM Quantum Platform account setup")
    print("(your API key will not be echoed to the screen or printed anywhere)\n")

    token = getpass.getpass("Paste your IBM Cloud API key: ").strip()
    if not token:
        raise SystemExit("No API key entered, aborting.")

    crn = input("Paste your instance CRN: ").strip()
    if not crn:
        raise SystemExit("No CRN entered, aborting.")

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
        overwrite=True,
        set_as_default=True,
    )
    print("\nAccount saved. You can now run: python run.py --real")


if __name__ == "__main__":
    main()
