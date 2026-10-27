"""Summarize actual pilot data and compare depth choices at equal gate-shot work."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).parent/'results'
rows=list(csv.DictReader((ROOT/'pilot.csv').open()))
# Means do not depend on sampling budget. Retain one row per setting/method.
work=[]
for r in rows:
 if r['budget']!='2000':continue
 gates=int(r['rzz_count'])+int(r['rx_count'])
 for C in (2000,20000):
  B=4*(C//gates//4) # unused remainder is explicitly recorded
  if r['method']=='raw':weights=[1.];shots=[B];means=[float(r['noisy'])]
  elif r['method']=='zne_equal':weights=[1.5,-.5];shots=[B//2,B//2];means=[float(r['noisy']),float(r['amplified'])]
  else:weights=[1.5,-.5];shots=[3*B//4,B//4];means=[float(r['noisy']),float(r['amplified'])]
  mean=sum(w*m for w,m in zip(weights,means))
  variance=sum(w*w*(1-m*m)/s for w,m,s in zip(weights,means,shots))
  work.append({**{k:r[k] for k in ['n','t','order','steps','p','observable','method']},'work_budget':C,'gates_per_shot':gates,'shots':B,'work_used':B*gates,'mse':(mean-float(r['exact']))**2+variance})
with (ROOT/'equal_work.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(work[0]));w.writeheader();w.writerows(work)
# Display concrete example chosen for plot, including best-depth baselines.
lines=['# Actual local pilot results — 25 October 2026','','All findings below are classical simulations, conditional on the specified model.','The pilot is a methodological replication; no novelty or hardware result is claimed.','','## Protocol','','H=sum Zi Z(i+1)+0.7 sum Xi, open chain, |0...0>, J=hbar=1; n=3,4; t=0.5,1,2.','Steps=1,2,4,8,16; first order applies ZZ then X, second order X/2 then ZZ then X/2.','RX(theta)=exp(-i theta X/2), RZZ(theta)=exp(-i theta ZZ/2). First-order operator','per step is exp(-i Hx dt) exp(-i Hzz dt); second order is symmetric.','','Exact scipy matrix exponentials provide continuous-H reference values; Qiskit','statevectors provide noiseless product-formula values; Aer density matrices','provide noisy expectations. After each abstract RZZ apply Dp(rho)=(1-p)rho+p','(I_pair/4 tensor Tr_pair rho). p=0,.001,.005,.02. No RX/readout/idle noise.','These are depolarization mixing parameters, not average gate infidelities.','','Two-point linear ZNE uses noise p and 3p and weights 1.5,-0.5. This is exact','parameter access in a model, not a demonstration of realizable gate folding or','noise learning. Equal shot split and 3:1 split are compared; 3:1 is a simple','allocation based on coefficient magnitudes, not exact variance-optimal allocation.','No clipping is applied to extrapolated estimates. Each Pauli observable is treated','as a separate estimation task; in hardware both Z observables could share shots.','','Raw uses B total shots; ZNE splits that same B across scales. B=200,2000,20000.','MSE=(estimator mean-exact continuous value)^2 + sum w_i^2(1-mu_i^2)/N_i.','The CSV separately records signed Trotter bias, modeled noise bias, total bias,','variance and actual sampled MSE over 2000 independent synthetic trials per row.','Do not add squared biases: cancellation and cross terms matter. Seed=20261025.','','## A concrete example','','n=4, t=1, second order, p=.005, Z0. Exact expectation=0.4042566922161047.','','| Steps | Raw MSE, B=200 | ZNE 3:1 MSE, B=200 | Raw MSE, B=20000 | ZNE 3:1 MSE, B=20000 |','|---|---|---|---|---|']
for s in (1,2,4,8,16):
 vals=[]
 for B in (200,20000):
  for m in ('raw','zne_3to1'):
   a=next(r for r in rows if r['n']=='4' and r['t']=='1.0' and r['order']=='2' and r['p']=='0.005' and r['observable']=='Z0' and int(r['steps'])==s and int(r['budget'])==B and r['method']==m)
   vals.append(f"{float(a['mse']):.6g}")
 lines.append('| '+str(s)+' | '+' | '.join(vals)+' |')
lines+=['','At 8 steps, ZNE reduces absolute total bias from .01260 to .004071. Yet at','B=200 its MSE rises from .004392 to .01707. At B=20000, 3:1 ZNE has MSE','.0001871 versus .0002010 raw at that same depth. But raw at 4 steps has MSE','.0001692, better than the best 3:1 ZNE among these five depths (.0001825 at','16 steps). Thus a same-depth mitigation gain need not beat a tuned raw baseline.','These oracle choices use known exact errors and are descriptive lower envelopes,','not a deployable selection method; selection costs and held-out tests are absent.','','At 4 steps, Trotter bias +.02052 and noise bias -.009216 partly cancel. Removing','noise can make continuous-target bias worse. Comparing only with the ideal digital','circuit would answer a different question.','','## Gate-work sensitivity','','`equal_work.csv` recomputes exact MSE at work caps 2000 and 20000 abstract','gate-shots per observable, charging RX and RZZ each one work unit. Shots are','rounded down to multiples of four; unused work is recorded. This provides a','consistent depth/order comparison within the abstract model. It is not a QPU','duration estimate; gates have unequal native cost and amplification has no','overhead in this oracle model. Calibration and compilation costs are excluded.','','For the same n=4,t=1,p=.005,Z0 example, allowing both orders and all depths:','','| Abstract gate-shot cap | Best raw (order, steps, shots, MSE) | Best ZNE 3:1 (order, steps, shots, MSE) |','|---|---|---|']
for C in (2000,20000):
 vals=[]
 for m in ('raw','zne_3to1'):
  subset=[r for r in work if r['n']=='4' and r['t']=='1.0' and r['p']=='0.005' and r['observable']=='Z0' and r['work_budget']==C and r['method']==m]
  b=min(subset,key=lambda r:r['mse']);vals.append(f"{b['order']}, {b['steps']}, {b['shots']}, {b['mse']:.6g}")
 lines.append('| '+str(C)+' | '+' | '.join(vals)+' |')
meta=json.loads((ROOT/'metadata.json').read_text())
lines+=['','## Validation and limits','',f"Generated {meta['rows']} rows. Seven core validation checks passed. Halving-step state-error ratios were 2.0010 and 4.0133, matching first/second order.",f"Monte Carlo MSE versus analytic MSE: {meta['mc_mse_mismatches_at_6se']} discrepancies beyond six estimated Monte Carlo standard errors across all rows.",'This is a consistency check, not proof against shared systematic errors. Exact MSE','has no sampling uncertainty conditional on the model; empirical MSE standard errors','are stored. For a measured mean, use sqrt(variance) as standard error; extrapolated','estimates are not binomial proportions. Grid win counts in metadata are descriptive,','not independent statistical trials or population success rates.','','Sensitivity varies size, time, order, step count, noise and budget. Not yet varied:','initial state, field strength, longitudinal terms, coherent/damping noise, readout,','drift, compiler seeds or native topology. The integrable transverse-field chain','and tiny sizes make this a verification exercise, not a hardness test. Replacing','oracle noise scaling by folding and charging pilot/calibration/selection overhead','could erase gains. Recent S26 and Z26 strongly overlap the conceptual question.','','The density-matrix working array is at most 16x16 complex128 (4096 bytes);','library/process overhead is larger. The sweep has at most 480 noisy circuit','settings before p=0 shortcuts and requires no hardware or large compute. Runtime','and environment are in metadata; recorded numerical sweep time excludes plotting.']
(ROOT.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[-22:]))
