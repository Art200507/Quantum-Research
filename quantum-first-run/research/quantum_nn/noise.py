"""Noise sensitivity of trained actual circuits; local Aer density matrices."""
from core import *
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,depolarizing_error

def main():
 review();log('06_plan',{'hypothesis':'Quantify whether native-basis modeled gate errors or finite-shot errors dominate our small trained models.','scope':'5 fitted seeds, 32 new attention episodes/seed, 64 new QNN examples/seed, p=.001,.01,.05 per CX, p/10 per U. Model p is mixing probability, not measured IBM infidelity.','resources':'1280 overlap and 320 QNN circuits compiled once to all-to-all U/CX; 4800 local density evolutions, 2 qubits; no hardware access.'})
 attention=json.loads((LOG/'04_result.json').read_text())['fits'];qfits=json.loads((LOG/'01_result.json').read_text())['fits']
 circuits=[];meta=[];group={}
 for fit in attention:
  seed=fit['seed'];p=np.array(fit['parameters']);Q,K,y=episodes(17000+seed,32);group['attention',seed]=(y,p)
  for i in range(32):
   for j in range(8):
    qc=circuit(K[i,j],p[:4],True,reupload_scaled=True);qc.compose(circuit(Q[i],p[:4],True,reupload_scaled=True).inverse(),inplace=True)
    circuits.append(qc);meta.append(('attention',seed,i,j))
 for fit in qfits:
  seed=fit['seed'];p=np.array(fit['parameters']);X=np.random.default_rng(18000+seed).uniform(-np.pi,np.pi,(64,2));y=(np.cos(X[:,0])*np.cos(X[:,1])>0).astype(int);group['QNN',seed]=(y,p)
  for i,x in enumerate(X):circuits.append(circuit(x,p[:2]));meta.append(('QNN',seed,i,0))
 isa=transpile(circuits,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
 counts=[dict(c.count_ops()) for c in isa];depths=[c.depth() for c in isa]
 for qc in isa:qc.save_density_matrix()
 rows=[];reference={}
 for noise in (0.,.001,.01,.05):
  model=NoiseModel()
  if noise:
   model.add_all_qubit_quantum_error(depolarizing_error(noise,2),['cx'])
   model.add_all_qubit_quantum_error(depolarizing_error(noise/10,1),['u'])
  sim=AerSimulator(method='density_matrix',noise_model=model,max_parallel_threads=1)
  result=sim.run(isa,seed_simulator=19000).result();values={key:np.zeros((32,8)) if key[0]=='attention' else np.zeros(64) for key in group}
  for i,(kind,seed,a,b) in enumerate(meta):
   rho=np.asarray(result.data(i)['density_matrix']);assert abs(np.trace(rho)-1)<1e-10
   if kind=='attention':values[kind,seed][a,b]=float(rho[0,0].real)
   else:values[kind,seed][a]=float(np.dot(rho.diagonal().real,[1,1,-1,-1]))
  for key,val in values.items():
   y,p=group[key];kind,seed=key
   pred=val.argmax(1) if kind=='attention' else (p[2]*val+p[3]>0)
   if noise==0:reference[key]=val.copy()
   rows.append({'model':kind,'seed':seed,'p_cx':noise,'exact_noisy_accuracy':float(np.mean(pred==y)),'mean_abs_expectation_shift':float(np.mean(abs(val-reference[key]))),'examples':len(y)})
 log('06_result',{'rows':rows,'compiled_circuits':len(isa),'cx_range':[min(c.get('cx',0) for c in counts),max(c.get('cx',0) for c in counts)],'depth_range':[min(depths),max(depths)],'noisy_evolutions':len(isa)*4,'decision':'Compare accuracy and expectation shifts with shot study; noise is modeled, exact density means exclude finite-shot error. Need larger new episodes before robust hardware claims. No IBM run warranted while classical task baseline remains stronger.'})

if __name__=='__main__':main()
