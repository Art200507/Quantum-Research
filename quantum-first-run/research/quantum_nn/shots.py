"""Fair shot-budget test for top-one quantum attention retrieval."""
from core import *

def choose(scores,r):return np.argmax(scores+r.uniform(0,1e-10,scores.shape),axis=1)
def adaptive(P,B,r,return_scores=False):
 n,m=P.shape;success=np.zeros((n,m));shots=np.zeros((n,m));active=np.tile(np.arange(m),(n,1));used=0
 for stage in range(3):
  count=active.shape[1];allocation=(B-used if stage==2 else B//3)//count;used+=allocation*count
  probs=np.take_along_axis(P,active,axis=1);hits=r.binomial(allocation,probs)
  row=np.arange(n)[:,None];success[row,active]+=hits;shots[row,active]+=allocation
  mean=success[row,active]/shots[row,active]+r.uniform(0,1e-10,(n,count))
  if stage<2:
   indices=np.argsort(mean,axis=1)[:,-count//2:];active=np.take_along_axis(active,indices,axis=1)
  else:pred=active[np.arange(n),mean.argmax(1)]
 assert used<=B
 if return_scores:
  final=np.full(P.shape,-np.inf);final[np.arange(n)[:,None],active]=success[np.arange(n)[:,None],active]/shots[np.arange(n)[:,None],active]
  return pred,used,final
 return pred,used

def main():
 review();log('05_plan',{'hypothesis':'Adaptive allocation may improve top-one attention retrieval at fixed total shots; include all screening shots.','scope':'Hard top-one sparse retrieval, not an unbiased estimator of all soft attention weights. Freeze experiment4 parameters. New episodes, five fitted seeds, 20 shot repeats per condition.'})
 fits=json.loads((LOG/'04_result.json').read_text())['fits'];rows=[];checks=[]
 for fit in fits:
  seed=fit['seed'];p=np.array(fit['parameters']);Q,K,y=episodes(15000+seed,512)
  P=overlaps(Q,K,p[:4],reupload_scaled=True);delta=(K-Q[:,None,:])**2;cp=np.array(fit['classical_parameters'])
  rows.append({'seed':seed,'model':'ideal_quantum','budget':0,'repeat':0,'accuracy':float(np.mean(P.argmax(1)==y))})
  rows.append({'seed':seed,'model':'classical_trained_metric','budget':0,'repeat':0,'accuracy':float(np.mean((-(delta[:,:,0]+np.exp(cp[0])*delta[:,:,1])).argmax(1)==y))})
  for B in (128,512,2048):
   for rep in range(20):
    r=np.random.default_rng(16000+seed*1000+B+rep)
    uniform=r.binomial(B//8,P)/(B//8);pred=choose(uniform,r)
    rows.append({'seed':seed,'model':'uniform','budget':B,'shots_used':(B//8)*8,'repeat':rep,'accuracy':float(np.mean(pred==y))})
    pred,used=adaptive(P,B,r)
    rows.append({'seed':seed,'model':'adaptive_halving','budget':B,'shots_used':used,'repeat':rep,'accuracy':float(np.mean(pred==y))})
  # Verify revised vectorized circuit map and compute-uncompute at fresh parameters.
  for i in range(3):
   qc=circuit(K[i,0],p[:4],True,reupload_scaled=True);qc.compose(circuit(Q[i],p[:4],True,reupload_scaled=True).inverse(),inplace=True)
   error=abs(abs(Statevector.from_instruction(qc).data[0])**2-P[i,0]);checks.append(error);assert error<1e-12
 log('05_result',{'rows':rows,'revised_circuit_max_error':max(checks),'decision':'Inspect fixed-budget gains and shot requirements; no quantum advantage follows because trained classical metric uses direct arithmetic without measurement shots.'})

if __name__=='__main__':main()
