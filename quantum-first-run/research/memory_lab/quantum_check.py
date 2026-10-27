"""Experiment 9: actual Qiskit circuits, ideal local statevectors only."""
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.circuit.library import StatePreparation,DiagonalGate
from qiskit.quantum_info import Statevector
from lab import data,save

def main():
 rows=[];maxerr=0;gatecounts=[];depths=[]
 for seed in range(100,112):
  for n in (2,4,8):
   K,V,y,r=data(seed,n,8,c=4)
   mem=np.sum(K*V[y],axis=0).astype(np.complex128);mem/=np.linalg.norm(mem)
   probs=[];signed=[]
   for q in K:
    unbinding=np.exp(-1j*np.angle(q.astype(np.complex128)))
    retrieved=mem*unbinding
    vals=[]
    for v in V:
     target=v.astype(np.complex128);target/=np.linalg.norm(target)
     qc=QuantumCircuit(3)
     qc.append(StatePreparation(mem),range(3))
     qc.append(DiagonalGate(unbinding),range(3))
     qc.append(StatePreparation(target).inverse(),range(3))
     actual=float(abs(Statevector.from_instruction(qc).data[0])**2)
     expected=float(abs(np.vdot(target,retrieved))**2)
     maxerr=max(maxerr,abs(actual-expected));assert abs(actual-expected)<1e-10
     isa=transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=23)
     gatecounts.append(int(isa.count_ops().get('cx',0)));depths.append(isa.depth())
     vals.append(actual)
    probs.append(vals)
    signed.append([(np.vdot(v.astype(np.complex128)/np.linalg.norm(v.astype(np.complex128)),retrieved)).real for v in V])
   probs=np.array(probs)
   for method,S,shots in [('classical_signed',np.array(signed),0),('quantum_exact_overlap',probs,0),('quantum_sampled_128',r.binomial(128,probs)/128,128),('quantum_sampled_2048',r.binomial(2048,probs)/2048,2048)]:
    rows.append({'seed':seed,'n':n,'d':8,'method':method,'shots_per_candidate':shots,'candidates':4,'total_shots_per_query':shots*4,'accuracy':float(np.mean(S.argmax(1)==y))})
 save(9,rows,{'qubits':3,'circuit_count':len(gatecounts),'max_probability_disagreement':maxerr,'mean_cx':float(np.mean(gatecounts)),'min_cx':min(gatecounts),'max_cx':max(gatecounts),'mean_depth':float(np.mean(depths)),'max_depth':max(depths),'sampling':'Independent binomial draws from exact circuit outcome probabilities, not hardware shots. No noise modeled.','cost':'Each candidate needs a fresh prepared memory; four candidates per query. Classical source mem and V and state-preparation instructions remain required; no QRAM assumed.','global_phase_check':'For any normalized v, |<v|v>|^2 = |<v|-v>|^2 = 1. Overlap probabilities cannot recover the signed score used by the classical model. This tests one encoding/readout, not all quantum memories.'})

if __name__=='__main__':main()
