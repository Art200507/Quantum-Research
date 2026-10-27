"""Experiment13: transfer measurement-allocation gains into frozen hybrid blocks."""
from core import *
from hybrid_block import dataset,forward
from shots import adaptive

def main():
 review();previous=json.loads((LOG/'12_result.json').read_text())
 log('13_plan',{'hypothesis':'Adaptive sparse readout may improve actual hybrid-block classification, but sparsifying attention changes the distribution seen by the trained FFN and need not help.','design':'Freeze validation-selected quantum/classical block parameters from12. Three seeds x256 new test episodes,8 keys; B128/512/2048,20 shot repeats. No additional calibration or parameter fitting. Uniform dense vs adaptive two-survivor sparse; same total shots but not same control latency or attention semantics.','optimizer_qualification':'Two quantum runs in12 hit the60-iteration limit; these checkpoints are provisional, not proven converged. No new optimization is concealed in this readout study.'})
 rows=[];differences=[]
 for fit in previous['fits']:
  seed=fit['seed'];kind=fit['kind'];p=np.array(fit['parameters']);ds=dataset(50000+seed,256);Q,K,v,y=ds
  logits=forward(ds,p,kind)
  rows.append({'seed':seed,'kind':kind,'method':'ideal','budget':0,'repeat':0,'accuracy':float(np.mean((logits>0)==y)),'cross_entropy':float(np.mean(np.logaddexp(0,logits)-y*logits))})
  if kind!='quantum':continue
  rows.append({'seed':seed,'kind':'control','method':'majority_values','budget':0,'repeat':0,'accuracy':float(np.mean((v.sum(1)>0)==y))})
  P=overlaps(Q,K,p[:4],reupload_scaled=True)
  for B in (128,512,2048):
   gains=[]
   for rep in range(20):
    r=np.random.default_rng(51000+seed*10000+B+rep)
    uniform=r.binomial(B//8,P)/(B//8)
    _,used,sparse=adaptive(P,B,r,return_scores=True);assert used==B
    acc={}
    for method,S,rounds in [('uniform_dense',uniform,1),('adaptive_sparse',sparse,3)]:
     logits=forward(ds,p,kind,S);acc[method]=float(np.mean((logits>0)==y))
     rows.append({'seed':seed,'kind':kind,'method':method,'budget':B,'rounds':rounds,'repeat':rep,'accuracy':acc[method],'cross_entropy':float(np.mean(np.logaddexp(0,logits)-y*logits))})
    gains.append(acc['adaptive_sparse']-acc['uniform_dense'])
   differences.append({'seed':seed,'budget':B,'mean_accuracy_gain':float(np.mean(gains))})
 log('13_result',{'rows':rows,'per_seed_gains':differences,'decision':'Inspect whether improvement survives complete-block output. Keep uniform-dense and sparse heads labeled distinctly; no energy or latency savings implied by equal-shot accuracy.'})

if __name__=='__main__':main()
