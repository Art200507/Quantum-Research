"""Test adaptive quantum-attention readout under actual simulated circuit noise."""
from core import *
from shots import adaptive,choose
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,depolarizing_error

def main():
 review();log('07_plan',{'hypothesis':'Even when depolarization preserves rank, compressed probability margins can damage finite-shot attention. Test whether adaptive measurements retain their benefit on fresh episodes.','budget':'5 trained seeds, 128 fresh episodes each, 8 candidates, ideal vs pCX=.05,pU=.005; 128/512 total shots; 20 synthetic measurement repeats; actual local noisy circuit density probabilities.','limits':'Adaptive rounds require extra classical feedback/control latency; matched shots is not matched wall time. No quantum hardware or full transformer claim.'})
 fits=json.loads((LOG/'04_result.json').read_text())['fits'];circuits=[];meta=[];ys={};rows=[];margin=[]
 for fit in fits:
  seed=fit['seed'];p=np.array(fit['parameters']);Q,K,y=episodes(21000+seed,128);ys[seed]=y
  for i in range(128):
   for j in range(8):
    qc=circuit(K[i,j],p[:4],True,reupload_scaled=True);qc.compose(circuit(Q[i],p[:4],True,reupload_scaled=True).inverse(),inplace=True)
    circuits.append(qc);meta.append((seed,i,j))
 isa=transpile(circuits,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
 for qc in isa:qc.save_density_matrix()
 for noise in (0.,.05):
  model=NoiseModel()
  if noise:
   model.add_all_qubit_quantum_error(depolarizing_error(noise,2),['cx']);model.add_all_qubit_quantum_error(depolarizing_error(noise/10,1),['u'])
  sim=AerSimulator(method='density_matrix',noise_model=model,max_parallel_threads=1)
  res=sim.run(isa,seed_simulator=22000).result();Ps={s:np.zeros((128,8)) for s in ys}
  for idx,(seed,i,j) in enumerate(meta):Ps[seed][i,j]=float(np.asarray(res.data(idx)['density_matrix'])[0,0].real)
  for seed,P in Ps.items():
   P=np.clip(P,0,1);y=ys[seed];sort=np.sort(P,axis=1)
   margin.append({'seed':seed,'pCX':noise,'mean_top_gap':float(np.mean(sort[:,-1]-sort[:,-2])),'exact_rank_accuracy':float(np.mean(P.argmax(1)==y))})
   for B in (128,512):
    for rep in range(20):
     r=np.random.default_rng(23000+seed*10000+B+rep+int(noise*100000))
     pred=choose(r.binomial(B//8,P)/(B//8),r)
     rows.append({'seed':seed,'pCX':noise,'budget':B,'repeat':rep,'method':'uniform','accuracy':float(np.mean(pred==y)),'shots_used':B})
     pred,used=adaptive(P,B,r)
     rows.append({'seed':seed,'pCX':noise,'budget':B,'repeat':rep,'method':'adaptive','accuracy':float(np.mean(pred==y)),'shots_used':used})
 # Independence unit is trained-seed+episode dataset. Repeats estimate measurement variability only.
 comparisons=[]
 for noise in (0.,.05):
  for B in (128,512):
   differences=[]
   for seed in ys:
    mean=lambda m:np.mean([r['accuracy'] for r in rows if r['pCX']==noise and r['budget']==B and r['seed']==seed and r['method']==m])
    differences.append(float(mean('adaptive')-mean('uniform')))
   comparisons.append({'pCX':noise,'budget':B,'seed_accuracy_differences':differences,'mean_gain':float(np.mean(differences)),'between_seed_se':float(np.std(differences,ddof=1)/np.sqrt(len(differences)))})
 log('07_result',{'rows':rows,'margins':margin,'paired_comparisons':comparisons,'density_evolutions':len(isa)*2,'decision':'Retain this as an engineering baseline for budgeted quantum attention if gain survives noise; investigate prior adaptive measurement work before novelty claims. Still compare to stronger classical trained metric; no advantage established.'})

if __name__=='__main__':main()
