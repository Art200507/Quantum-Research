"""Experiment12: minimal trainable hybrid cross-attention Transformer block.

Fixed token embedding/projections; attention -> residual/LN -> learned tanh FFN
-> residual/LN -> binary readout. Quantum or classical head, 50 parameters each.
Not an LLM and not a reproduction of QASA.
"""
from core import *

def dataset(seed,n):
 Q,K,idx=episodes(seed,n,8);v=np.random.default_rng(seed+1000).choice([-1.,1.],size=(n,8))
 y=(v[np.arange(n),idx]>0).astype(float)
 return Q,K,v,y

def norm(x):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)

def forward(ds,p,kind,shot_estimates=None):
 Q,K,v,y=ds;n=len(Q);beta=np.exp(p[4]);delta=K-Q[:,None,:]
 if kind=='quantum':
  S=overlaps(Q,K,p[:4],reupload_scaled=True) if shot_estimates is None else shot_estimates
 else:S=-np.sum((delta@p[:4].reshape(2,2))**2,axis=2)
 attention=softmax(beta*S,axis=1)
 tokens=np.concatenate([K,v[:,:,None],np.zeros((n,8,1))],axis=2)
 query=np.c_[Q,np.zeros(n),np.ones(n)]
 z=norm(query+np.einsum('nm,nmd->nd',attention,tokens))
 W1=p[5:21].reshape(4,4);b1=p[21:25];W2=p[25:41].reshape(4,4);b2=p[41:45]
 z=norm(z+np.tanh(z@W1+b1)@W2+b2)
 return z@p[45:49]+p[49]

def main():
 review();previous=json.loads((LOG/'11_result.json').read_text())
 log('12_plan',{'hypothesis':'Test the quantum head inside a complete small hybrid cross-attention block, rather than assuming isolated-head gains transfer through residuals, normalization and an FFN.','architecture':'4-dimensional tokens [key0,key1,value,role], fixed identity value/output maps, attention+residual+fixed LayerNorm, trainable4x4 tanh FFN+residual+fixed LayerNorm, binary readout. Quantum4 parameters+temperature or classical2x2 metric+temperature; 50 trainable parameters per model.','design':'3 seeds,64 train/64 validation/256 test episodes; warm-start both heads from prior trained retrieval maps; shared classical initialization within seed. Joint exact-expectation L-BFGS-B up to60 iterations, checkpoint selected on validation loss. Evaluate ideal and finite-shot quantum attention.','limits':'Warm-start transfer study, not from-scratch architecture comparison. Fixed projections/nontrainable normalization differ from standard Transformer. Classical optimizer steps do not equate wall-time or QPU training cost.','previous_rows_read':len(previous['rows'])})
 old=json.loads((LOG/'04_result.json').read_text())['fits'];rows=[];fits=[];equivalence=[]
 for seed in range(3):
  train=dataset(40000+seed,64);val=dataset(41000+seed,64);test=dataset(42000+seed,256)
  r=np.random.default_rng(43000+seed);shared=np.zeros(45)
  shared[:16]=r.normal(0,.08,16);shared[20:36]=r.normal(0,.08,16);shared[40:44]=r.normal(0,.1,4)
  for kind in ('quantum','classical'):
   head=np.array(old[seed]['parameters']).copy()
   if kind=='classical':head[:4]=np.diag([1,np.sqrt(np.exp(old[seed]['classical_parameters'][0]))]).ravel();head[4]=old[seed]['classical_parameters'][1]
   init=np.r_[head,shared];assert len(init)==50
   def loss(p,ds=train):
    logits=forward(ds,p,kind);y=ds[3]
    return float(np.mean(np.logaddexp(0,logits)-y*logits)+1e-4*np.mean(p[5:]**2))
   best={'params':init.copy(),'loss':loss(init,val),'iteration':0};counter=[0]
   def callback(p):
    counter[0]+=1;vl=loss(p,val)
    if vl<best['loss']:best.update(params=p.copy(),loss=vl,iteration=counter[0])
   bounds=([(.02,2),(.02,2),(-np.pi,np.pi),(-np.pi,np.pi)] if kind=='quantum' else [(-3,3)]*4)+[(0,np.log(160))]+[(-8,8)]*45
   start=time.perf_counter();fit=minimize(loss,init,method='L-BFGS-B',bounds=bounds,callback=callback,options={'maxiter':60,'ftol':1e-8,'maxfun':15000})
   p=best['params'];elapsed=time.perf_counter()-start
   fits.append({'seed':seed,'kind':kind,'parameters':p,'optimizer_success':bool(fit.success),'message':str(fit.message),'iterations':fit.nit,'calls':fit.nfev,'best_validation_iteration':best['iteration'],'validation_loss':best['loss'],'seconds':elapsed})
   for split,ds in [('train',train),('validation',val),('test',test)]:
    logits=forward(ds,p,kind);y=ds[3]
    rows.append({'seed':seed,'kind':kind,'split':split,'shots_per_query':0,'repeat':0,'accuracy':float(np.mean((logits>0)==y)),'cross_entropy':float(np.mean(np.logaddexp(0,logits)-y*logits))})
   if kind=='quantum':
    Q,K,v,y=test;P=overlaps(Q,K,p[:4],reupload_scaled=True)
    qc=circuit(K[0,0],p[:4],True,reupload_scaled=True);qc.compose(circuit(Q[0],p[:4],True,reupload_scaled=True).inverse(),inplace=True)
    error=abs(float(abs(Statevector.from_instruction(qc).data[0])**2)-P[0,0]);assert error<1e-12;equivalence.append(error)
    for B in (512,2048):
     for rep in range(10):
      S=np.random.default_rng(44000+seed*10000+B+rep).binomial(B//8,P)/(B//8)
      logits=forward(test,p,kind,S)
      rows.append({'seed':seed,'kind':'quantum_uniform_shots','split':'test','shots_per_query':B,'repeat':rep,'accuracy':float(np.mean((logits>0)==y)),'cross_entropy':float(np.mean(np.logaddexp(0,logits)-y*logits))})
   print('Completed',seed,kind,'test accuracy',rows[-1]['accuracy'],flush=True)
 log('12_result',{'rows':rows,'fits':fits,'trained_circuit_errors':equivalence,'decision':'Inspect test and convergence records. This establishes an executable small hybrid block, not a full language transformer or quantum advantage. Any optimizer budget failures remain explicit.'})

if __name__=='__main__':main()
