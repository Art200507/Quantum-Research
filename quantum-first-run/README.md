# quantum-first-run

Local demonstrations of bit-frequency measurements, Bell/GHZ populations, and
mirror-circuit return probability. These are diagnostics, not evidence of
quantum advantage, certified randomness, or measured entanglement.

Use the existing pinned environment: `.venv/bin/python run.py --shots 256 --mirror-sizes 3`.
The default FakeTorino path is local noisy simulation; large mirrors are only
compiled. Runtime is not instant and depends on simulation method and size.

Real submissions still require both `--real` and typing `yes`. Do not submit
hardware jobs for the current investigation. Setup is through `setup_account.py`;
it hides the API key but the CRN prompt is visible. Never commit credentials.

`--out DIRECTORY` now places JSON and plots together. Historical outputs in
`results/` are unchanged. The snapshot is simulated data, not hardware data.
Batching does not establish low cost. Old 30–90 second usage estimates were not
validated and are withdrawn. Current IBM plan details must be checked before use:
https://quantum.cloud.ibm.com/docs/en/guides/plans-overview

Mirror circuits have a known ideal output for every size. Generic rotations are
not stabilizer circuits, but fixed-depth chain structure supports classical
methods beyond full statevectors. Return probability includes readout effects.
Z-only Bell/GHZ histograms cannot distinguish a coherent entangled state from
an incoherent mixture. QRNG chi-square tests assess marginal balance under IID
assumptions, use asymptotic p-values, and do not certify unpredictability.

See [research/README.md](research/README.md) for the audit, literature, proposed
questions, local replication, results, learning roadmap and concept note.
Pre-correction source copies are in `research/originals/`.
