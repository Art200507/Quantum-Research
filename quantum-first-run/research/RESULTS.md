# Actual local pilot results — 25 October 2026

All findings below are classical simulations, conditional on the specified model.
The pilot is a methodological replication; no novelty or hardware result is claimed.

## Protocol

H=sum Zi Z(i+1)+0.7 sum Xi, open chain, |0...0>, J=hbar=1; n=3,4; t=0.5,1,2.
Steps=1,2,4,8,16; first order applies ZZ then X, second order X/2 then ZZ then X/2.
RX(theta)=exp(-i theta X/2), RZZ(theta)=exp(-i theta ZZ/2). First-order operator
per step is exp(-i Hx dt) exp(-i Hzz dt); second order is symmetric.

Exact scipy matrix exponentials provide continuous-H reference values; Qiskit
statevectors provide noiseless product-formula values; Aer density matrices
provide noisy expectations. After each abstract RZZ apply Dp(rho)=(1-p)rho+p
(I_pair/4 tensor Tr_pair rho). p=0,.001,.005,.02. No RX/readout/idle noise.
These are depolarization mixing parameters, not average gate infidelities.

Two-point linear ZNE uses noise p and 3p and weights 1.5,-0.5. This is exact
parameter access in a model, not a demonstration of realizable gate folding or
noise learning. Equal shot split and 3:1 split are compared; 3:1 is a simple
allocation based on coefficient magnitudes, not exact variance-optimal allocation.
No clipping is applied to extrapolated estimates. Each Pauli observable is treated
as a separate estimation task; in hardware both Z observables could share shots.

Raw uses B total shots; ZNE splits that same B across scales. B=200,2000,20000.
MSE=(estimator mean-exact continuous value)^2 + sum w_i^2(1-mu_i^2)/N_i.
The CSV separately records signed Trotter bias, modeled noise bias, total bias,
variance and actual sampled MSE over 2000 independent synthetic trials per row.
Do not add squared biases: cancellation and cross terms matter. Seed=20261025.

## A concrete example

n=4, t=1, second order, p=.005, Z0. Exact expectation=0.4042566922161047.

| Steps | Raw MSE, B=200 | ZNE 3:1 MSE, B=200 | Raw MSE, B=20000 | ZNE 3:1 MSE, B=20000 |
|---|---|---|---|---|
| 1 | 0.124411 | 0.133605 | 0.122274 | 0.125003 |
| 2 | 0.0100216 | 0.0224073 | 0.00622599 | 0.00717622 |
| 4 | 0.00426433 | 0.0170306 | 0.000169154 | 0.000577573 |
| 8 | 0.00439165 | 0.0170736 | 0.000200973 | 0.000187144 |
| 16 | 0.00540561 | 0.0174589 | 0.00113757 | 0.000182493 |

At 8 steps, ZNE reduces absolute total bias from .01260 to .004071. Yet at
B=200 its MSE rises from .004392 to .01707. At B=20000, 3:1 ZNE has MSE
.0001871 versus .0002010 raw at that same depth. But raw at 4 steps has MSE
.0001692, better than the best 3:1 ZNE among these five depths (.0001825 at
16 steps). Thus a same-depth mitigation gain need not beat a tuned raw baseline.
These oracle choices use known exact errors and are descriptive lower envelopes,
not a deployable selection method; selection costs and held-out tests are absent.

At 4 steps, Trotter bias +.02052 and noise bias -.009216 partly cancel. Removing
noise can make continuous-target bias worse. Comparing only with the ideal digital
circuit would answer a different question.

## Gate-work sensitivity

`equal_work.csv` recomputes exact MSE at work caps 2000 and 20000 abstract
gate-shots per observable, charging RX and RZZ each one work unit. Shots are
rounded down to multiples of four; unused work is recorded. This provides a
consistent depth/order comparison within the abstract model. It is not a QPU
duration estimate; gates have unequal native cost and amplification has no
overhead in this oracle model. Calibration and compilation costs are excluded.

For the same n=4,t=1,p=.005,Z0 example, allowing both orders and all depths:

| Abstract gate-shot cap | Best raw (order, steps, shots, MSE) | Best ZNE 3:1 (order, steps, shots, MSE) |
|---|---|---|
| 2000 | 1, 2, 140, 0.00845071 | 1, 2, 140, 0.0268583 |
| 20000 | 1, 4, 712, 0.00154854 | 1, 2, 1428, 0.00434261 |

## Validation and limits

Generated 4320 rows. Seven core validation checks passed. Halving-step state-error ratios were 2.0010 and 4.0133, matching first/second order.
Monte Carlo MSE versus analytic MSE: 0 discrepancies beyond six estimated Monte Carlo standard errors across all rows.
This is a consistency check, not proof against shared systematic errors. Exact MSE
has no sampling uncertainty conditional on the model; empirical MSE standard errors
are stored. For a measured mean, use sqrt(variance) as standard error; extrapolated
estimates are not binomial proportions. Grid win counts in metadata are descriptive,
not independent statistical trials or population success rates.

Sensitivity varies size, time, order, step count, noise and budget. Not yet varied:
initial state, field strength, longitudinal terms, coherent/damping noise, readout,
drift, compiler seeds or native topology. The integrable transverse-field chain
and tiny sizes make this a verification exercise, not a hardness test. Replacing
oracle noise scaling by folding and charging pilot/calibration/selection overhead
could erase gains. Recent S26 and Z26 strongly overlap the conceptual question.

The density-matrix working array is at most 16x16 complex128 (4096 bytes);
library/process overhead is larger. The sweep has at most 480 noisy circuit
settings before p=0 shortcuts and requires no hardware or large compute. Runtime
and environment are in metadata; recorded numerical sweep time excludes plotting.
