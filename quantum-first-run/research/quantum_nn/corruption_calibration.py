"""Experiment31: reliability under representative noisy calibration, with ambiguity cost."""
from prediction_sets import *

def main():
 review();previous=json.loads((LOG/'30_result.json').read_text())
 log('31_plan',{'hypothesis':'Calibration representative of corrupted deployment may restore prediction-set coverage by producing more ambiguous answers, without recovering lost classification information.','design':'Freeze selected models. Gaussian sigma.8 corruption of observed inputs, clean-input labels.5 fitted seeds x4 tasks x10 independently generated calibration/test cohorts,256 calibration and512 test. Quantum1024-shot grouped measurement both calibration/test; selected classical control too.','cost':'256*1024 calibration shots per fitted quantum predictor in each replicate;10 calibration replicates are research overhead, not a free deployment procedure.','limit':'This assumes the corruption distribution is represented in calibration. It does not solve unknown future shifts. Sets containing both labels are explicitly counted as ambiguity.','previous_result_rows_read':len(previous['rows'])})
 rows=[];selected=json.loads((LOG/'29_result.json').read_text())['selected']
 for item in selected:
  seed,task=item['seed'],item['task'];qfit=item['quantum_fit'];cfit=item['classical_fit']
  for rep in range(10):
   r=np.random.default_rng(100000+seed*100+rep);Xc=r.uniform(-np.pi,np.pi,(256,2));Xt=r.uniform(-np.pi,np.pi,(512,2));yc=labels(Xc,task);yt=labels(Xt,task)
   Xc=Xc+r.normal(0,.8,Xc.shape);Xt=Xt+r.normal(0,.8,Xt.shape)
   cq=cutoff(sampled_z(Xc,qfit,101000+seed*100+rep),yc);ct=cutoff(classical_predict(Xc,cfit),yc)
   for model,z,q in [('quantum_measured',sampled_z(Xt,qfit,102000+seed*100+rep),cq),('classical',classical_predict(Xt,cfit),ct)]:rows.append({'seed':seed,'task':task,'model':model,'replicate':rep,'cutoff':q,**set_metrics(z,yt,q)})
 log('31_result',{'rows':rows,'decision':'A coverage recovery bought with many ambiguous answers is useful uncertainty handling, not improved discrimination. Next preserve an explicit classical representation control for the richer quantum features.'})

if __name__=='__main__':main()
