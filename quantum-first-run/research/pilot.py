"""Small, entirely local replication. No service, account, or hardware imports."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import time
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

OUT = Path(__file__).parent / 'results'
SEED = 20261025

def pauli(n, sites):
    chars = ['I'] * n
    for q, char in sites.items():
        chars[n - 1 - q] = char
    return SparsePauliOp(''.join(chars)).to_matrix()

def hamiltonian(n, h=.7):
    return sum(pauli(n, {q:'Z',q+1:'Z'}) for q in range(n-1)) + h * sum(pauli(n,{q:'X'}) for q in range(n))

def circuit(n, t, steps, order, h=.7):
    qc = QuantumCircuit(n)
    dt = t / steps
    for _ in range(steps):
        if order == 2:
            for q in range(n): qc.rx(h*dt,q)
        for q in range(n-1): qc.rzz(2*dt,q,q+1)
        for q in range(n): qc.rx(2*h*dt if order == 1 else h*dt,q)
    return qc

def density(qc, p):
    c = qc.copy()
    c.save_density_matrix()
    model = NoiseModel()
    if p:
        model.add_all_qubit_quantum_error(depolarizing_error(p,2), ['rzz'])
    # Preserve abstract RX/RZZ gates. Noise is per RZZ, not a native-device model.
    sim = AerSimulator(method='density_matrix', noise_model=model, basis_gates=['rx','rzz'], max_parallel_threads=1)
    return np.asarray(sim.run(c, seed_simulator=SEED).result().data(0)['density_matrix'])

def checks():
    n=3; z=pauli(n,{0:'Z'}); psi=np.eye(2**n,dtype=complex)[:,0]
    assert np.allclose(hamiltonian(n), hamiltonian(n).conj().T)
    assert np.allclose(Statevector.from_instruction(circuit(n,0,2,2)).data,psi)
    qc=circuit(n,.8,4,2); v=Statevector.from_instruction(qc).data
    assert np.allclose(density(qc,0),np.outer(v,v.conj()),atol=1e-12)
    rho=density(qc,.02)
    assert np.isclose(np.trace(rho),1) and np.linalg.eigvalsh(rho).min()>-1e-12
    # Analytic depolarization of |00> after a diagonal RZZ: <Z0> = 1-p.
    two=QuantumCircuit(2);two.rzz(.3,0,1)
    assert np.isclose(np.trace(density(two,.1) @ pauli(2,{0:'Z'})).real,.9)
    errors={}
    exact=expm(-1j*hamiltonian(n)*.8)@psi
    for order in (1,2):
        errs=[np.linalg.norm(Statevector.from_instruction(circuit(n,.8,s,order)).data-exact) for s in (8,16)]
        errors[order]=errs[0]/errs[1]
        assert (1.8 < errors[order] < 2.2) if order==1 else (3.7 < errors[order] < 4.3)
    # Qubit indexing checked against an asymmetric computational basis state.
    v=np.zeros(8);v[1]=1
    assert np.isclose(v@z@v,-1) and np.isclose(v@pauli(3,{2:'Z'})@v,1)
    return {'passed':True,'state_error_halving_ratios':errors,'checks':['Hermiticity','zero time','Aer versus statevector','trace and positivity','analytic depolarization','first/second-order convergence','little-endian observable indexing']}

def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=True)
    validation=checks(); rows=[]; rng=np.random.default_rng(SEED)
    trials=2000
    # 2 sizes x 3 times x 2 orders x 5 step counts x 4 p x 2 noise scales.
    # Largest density matrix is 16x16 (4096 bytes complex128).
    for n in (3,4):
      psi=np.eye(2**n,dtype=complex)[:,0]; H=hamiltonian(n)
      obs={'Z0':pauli(n,{0:'Z'}),'Z0Z1':pauli(n,{0:'Z',1:'Z'})}
      for t in (.5,1.,2.):
       exact=expm(-1j*H*t)@psi
       for order in (1,2):
        for steps in (1,2,4,8,16):
         qc=circuit(n,t,steps,order); ideal=Statevector.from_instruction(qc).data
         for p in (0.,.001,.005,.02):
          rhos=[density(qc,scale*p) for scale in (1,3)] if p else [np.outer(ideal,ideal.conj())]*2
          for name,O in obs.items():
           truth=float(np.vdot(exact,O@exact).real); trotter=float(np.vdot(ideal,O@ideal).real)
           means=[float(np.clip(np.trace(r@O).real,-1,1)) for r in rhos]
           for B in (200,2000,20000):
            for method, weights, shots in [('raw',[1.],[B]),('zne_equal',[1.5,-.5],[B//2,B//2]),('zne_3to1',[1.5,-.5],[3*B//4,B//4])]:
             mus=means[:len(weights)]; mean=float(np.dot(weights,mus))
             variance=float(sum(w*w*(1-m*m)/s for w,m,s in zip(weights,mus,shots)))
             estimates=sum(w*(2*rng.binomial(s,(1+m)/2,size=trials)/s-1) for w,m,s in zip(weights,mus,shots))
             squared=(estimates-truth)**2
             mse=(mean-truth)**2+variance
             rows.append(dict(n=n,t=t,order=order,steps=steps,p=p,observable=name,budget=B,method=method,rzz_count=(n-1)*steps,rx_count=n*steps*(2 if order==2 else 1),abstract_depth=qc.depth(),exact=truth,trotter=trotter,noisy=means[0],amplified=means[1],estimator_mean=mean,trotter_bias=trotter-truth,noise_bias=means[0]-trotter,total_bias=mean-truth,variance=variance,mse=mse,empirical_mse=float(squared.mean()),empirical_mse_se=float(squared.std(ddof=1)/np.sqrt(trials)),shots_by_scale=json.dumps(shots)))
    with (OUT/'pilot.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    wins={}
    for B in (200,2000,20000):
        groups={}
        for row in rows:
            if row['budget']==B:
                key=tuple(row[k] for k in ('n','t','order','steps','p','observable'))
                groups.setdefault(key,{})[row['method']]=row['mse']
        wins[B]={m:sum(g[m]<g['raw'] for g in groups.values()) for m in ('zne_equal','zne_3to1')}
        wins[B]['settings']=len(groups)
    # Flag mismatch between Monte Carlo and exact estimator MSE, allowing finite MC error.
    mismatches=sum(abs(r['empirical_mse']-r['mse'])>6*r['empirical_mse_se']+1e-12 for r in rows)
    meta={'seed':SEED,'trials_per_estimator':trials,'rows':len(rows),'seconds':time.perf_counter()-start,'python':platform.python_version(),'platform':platform.platform(),'dependencies':{p:metadata.version(p) for p in ['numpy','scipy','qiskit','qiskit-aer','qiskit-ibm-runtime','matplotlib']},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'validation':validation,'mc_mse_mismatches_at_6se':mismatches,'zne_wins_at_fixed_steps':wins,'model':{'H':'sum Z_i Z_(i+1) + 0.7 sum X_i','boundary':'open','initial_state':'all zero','hbar':1,'J':1,'noise':'two-qubit depolarizing channel after each abstract RZZ only','scales':[1,3],'amplification':'oracle parameter scaling p -> 3p, not gate folding','budgets':'total shots per observable; not equal QPU duration','hardware':False}}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    plot(rows)

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for ax,B in zip(axes,(200,20000)):
      for method in ('raw','zne_equal','zne_3to1'):
        selected=[r for r in rows if r['n']==4 and r['t']==1 and r['order']==2 and r['p']==.005 and r['observable']=='Z0' and r['budget']==B and r['method']==method]
        ax.plot([r['steps'] for r in selected],[r['mse'] for r in selected],'o-',label=method)
      ax.set(xscale='log',yscale='log',xlabel='Trotter steps',ylabel='MSE versus exact continuous evolution',title=f'Total shots per observable = {B}')
      ax.legend()
    fig.suptitle('Local model only: 4 spins, t=1, second order, p=0.005')
    fig.tight_layout();fig.savefig(OUT/'mse.png',dpi=160);plt.close(fig)

if __name__=='__main__': main()
