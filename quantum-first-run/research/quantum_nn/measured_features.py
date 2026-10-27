"""Joint measurement simulation: shared-outcome covariance, exact shot ledger."""
from feature_readout import *
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
ROT={'Z':np.eye(2),'X':H,'Y':H@np.diag([1,-1j])}
SIGNS={'I':np.ones(4),'q0':np.array([1,-1,1,-1]),'q1':np.array([1,1,-1,-1])}

def basis_probabilities(X,frequencies,bases):
 result=[]
 for f in frequencies:
  S=map_states(X,f)
  for basis in bases:
   R=apply(apply(S,ROT[basis[1]],0),ROT[basis[0]],1)
   P=abs(R)**2;P/=P.sum(1,keepdims=True);result.append(P)
 return result

def reconstruct_exact(probabilities,frequencies,bases):
 return measure(probabilities,frequencies,bases,None,None)[0]

def measure(probabilities,frequencies,bases,budget,r):
 n=len(probabilities[0]);F=np.zeros((n,15*len(frequencies)));den=np.zeros(15*len(frequencies));groups=len(probabilities)
 allocations=[1]*groups if budget is None else [budget//groups+int(i<budget%groups) for i in range(groups)]
 if min(allocations)<1:raise ValueError('Budget too small')
 for i,(P,N) in enumerate(zip(probabilities,allocations)):
  counts=P if budget is None else r.multinomial(N,P)
  view=i//len(bases);basis=bases[i%len(bases)]
  for j,pauli in enumerate(PAULIS):
   if any(p!='I' and p!=b for p,b in zip(pauli,basis)):continue
   sign=(SIGNS['q1'] if pauli[0]!='I' else SIGNS['I'])*(SIGNS['q0'] if pauli[1]!='I' else SIGNS['I'])
   F[:,view*15+j]+=counts@sign;den[view*15+j]+=N
 covered=den>0;F[:,covered]/=den[covered]
 return F,sum(allocations) if budget is not None else 0,covered

def metrics(z,y):
 p=expit(z);correct=(p>.5)==y;conf=np.maximum(p,1-p)
 return {**evaluate(z,y),'confident_wrong_fraction':float(np.mean((conf>=.9)&~correct))}

def main(stage):
 review();bases=[a+b for a in 'XYZ' for b in 'XYZ'] if stage==27 else ['XX','ZZ','XZ','YY']
 log(f'{stage}_plan',{'hypothesis':'Ideal multi-output capability may collapse when all quantum observables share a finite total measurement budget.' if stage==27 else 'Measure only the bases needed by retained observables; compare against the same-budget full9-basis protocol, retaining correlations.','design':'Frozen feature-readout models from25/26;5 seeds,4 tasks,512 fresh test points, budgets256/1024/4096,10 measurement repeats.','measurement':'Actual joint categorical Born outcomes for each two-qubit basis; all Pauli features from a setting share those counts. Total shots exactly distributed over views/bases including remainders. No independent-binomial fiction for jointly measured features.','bases':bases,'training':'Ideal-training readouts unchanged. No gate noise here. Classical computation of two-qubit Born probabilities independently validated against circuit feature expectations.'})
 rows=[];checks=[]
 X=np.random.default_rng(85000).uniform(-np.pi,np.pi,(10,2));freq=[.5,1.,2.,3.];probs=basis_probabilities(X,freq,bases);F,_,covered=measure(probs,freq,bases,None,None);truth=features(X,freq)
 error=float(np.max(abs(F[:,covered]-truth[:,covered])));assert error<1e-12;checks.append(error)
 for old_stage in (25,26):
  fits=[f for f in json.loads((LOG/f'{old_stage}_result.json').read_text())['fits'] if f['model']=='quantum_features']
  # Same fresh inputs and random streams across tasks; outcomes reused across tasks for honest paired analysis.
  for seed in range(5):
   X=np.random.default_rng(86000+seed).uniform(-np.pi,np.pi,(512,2));freq=fits[0]['frequencies'];probs=basis_probabilities(X,freq,bases)
   cohort=[f for f in fits if f['seed']==seed]
   for f in cohort:
    mask=np.array(f['mask'],bool);assert np.all(covered.reshape(4,15)[0][np.flatnonzero(mask)%15]) if stage==28 else True
    y=labels(X,f['task']);rows.append({'seed':seed,'task':f['task'],'source_stage':old_stage,'budget':0,'repeat':0,'model':'ideal','basis_settings':len(bases)*len(freq),**metrics(readout(X,f),y)})
   for B in (256,1024,4096):
    for rep in range(10):
     F,used,cov=measure(probs,freq,bases,B,np.random.default_rng(87000+seed*10000+B+rep));assert used==B
     for f in cohort:
      assert np.all(cov[np.array(f['mask'],bool)])
      y=labels(X,f['task']);rows.append({'seed':seed,'task':f['task'],'source_stage':old_stage,'budget':B,'repeat':rep,'model':'measured','basis_settings':len(bases)*len(freq),**metrics(readout(X,f,F),y)})
 log(f'{stage}_result',{'rows':rows,'measurement_reconstruction_errors':checks,'decision':'Inspect both retained accuracy and confidence errors; reduced setting count does not by itself prove reliability or quantum superiority.'})

if __name__=='__main__':
 import sys
 main(int(sys.argv[1]))
