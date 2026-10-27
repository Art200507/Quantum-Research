"""Check actual quantum attention value outputs, not just the winning index."""
from core import *
from shots import adaptive

def main():
 review();log('08_plan',{'hypothesis':'A top-one retrieval gain may not transfer to attention weighted-value outputs. Evaluate the quantum attention operation softmax(beta*fidelity) @ V directly and calibrate beta on separate validation episodes.','scope':'5 fixed trained maps, 80 validation/256 test episodes per seed, 8 keys and 4-dimensional random values, B512/query,20 measurement repeats. Noisy shot estimates here mean finite shots from ideal probabilities, not gate noise.','calibration_cost':'80*512*20 = 819200 validation shots per seed per allocation method; all beta choices reuse those samples. This overhead is not free and must be amortized before claiming resource savings.'})
 fits=json.loads((LOG/'04_result.json').read_text())['fits'];rows=[];tuning=[];grid=np.array([5,10,20,40,80,160,320])
 for fit in fits:
  seed=fit['seed'];p=np.array(fit['parameters']);beta=np.exp(p[4]);best={}
  for split,N,data_seed in [('validation',80,28000+seed),('test',256,29000+seed)]:
   Q,K,y=episodes(data_seed,N);P=overlaps(Q,K,p[:4],reupload_scaled=True)
   V=np.random.default_rng(data_seed+2000).normal(size=(N,8,4));truth=V[np.arange(N),y]
   aggregate=lambda S,b:np.einsum('nm,nmd->nd',softmax(S*b,axis=1),V)
   if split=='test':
    for name,pred in [('ideal_quantum_soft',aggregate(P,beta)),('ideal_quantum_top1',V[np.arange(N),P.argmax(1)])]:
     rows.append({'seed':seed,'method':name,'beta':beta,'repeat':0,'mse_to_target_value':float(np.mean((pred-truth)**2))})
    delta=(K-Q[:,None,:])**2;score=-(delta[:,:,0]+np.exp(fit['classical_parameters'][0])*delta[:,:,1]);pred=V[np.arange(N),score.argmax(1)]
    rows.append({'seed':seed,'method':'trained_classical_top1','beta':0,'repeat':0,'mse_to_target_value':float(np.mean((pred-truth)**2))})
   estimates={'uniform':[],'adaptive_sparse':[]}
   for rep in range(20):
    r=np.random.default_rng(data_seed+30000+rep)
    estimates['uniform'].append(r.binomial(64,P)/64)
    _,_,S=adaptive(P,512,r,return_scores=True);estimates['adaptive_sparse'].append(S)
   for method,Ss in estimates.items():
    if split=='validation':
     losses=[float(np.mean([np.mean((aggregate(S,b)-truth)**2) for S in Ss])) for b in grid]
     best[method]=float(grid[np.argmin(losses)]);tuning.append({'seed':seed,'method':method,'grid':grid,'validation_mses':losses,'selected_beta':best[method]})
    else:
     for label,b in [('original',beta),('validation_calibrated',best[method])]:
      for rep,S in enumerate(Ss):
       pred=aggregate(S,b);ideal=aggregate(P,b)
       rows.append({'seed':seed,'method':method+'_'+label,'beta':b,'repeat':rep,'mse_to_target_value':float(np.mean((pred-truth)**2)),'mse_to_exact_same_beta_attention':float(np.mean((pred-ideal)**2))})
 log('08_result',{'rows':rows,'tuning':tuning,'decision':'Use value-output MSE as well as top-one recall for transformer-related work. Calibration overhead and sparse-vs-dense semantics must be stated. Retain only effects observed on independent test episodes; no full transformer or language benchmark evaluated.'})

if __name__=='__main__':main()
