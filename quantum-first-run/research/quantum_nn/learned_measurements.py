"""Experiment38: finite physical measurement of frozen learned encoders."""
from learned_inputs import *

def frozen_models():
 originals=json.loads((LOG/'35_result.json').read_text())['fits'];refined=json.loads((LOG/'37_result.json').read_text())['fits'];out=[]
 for f in originals:
  learned=f['learned'];match=[r for r in refined if r['seed']==f['seed'] and r['task']==f['task']]
  if match:learned=match[0]['refined']
  out.append({'seed':f['seed'],'task':f['task'],'baseline':f['baseline'],'learned':learned})
 return out

def main():
 review();log('38_plan',{'hypothesis':'Ideal encoder-learning gains may not survive finite measurements. Test frozen,validation-selected encoders with correct shared-basis outcome covariance.','design':'Models35/refinements37 frozen;5 seeds,4 tasks,512 new test points,1024/4096 total shots per prediction,10 measurement repeats. Four local bases per encoding scale.','fairness':'Same total shots for fixed/learned encoders,including all views and bases. No gate noise; ideal training may exploit features that are hard to measure.','selection':'All checkpoint choices already fixed; no use of this test for model tuning.'})
 rows=[];equivalence=[]
 for f in frozen_models():
  seed,task=f['seed'],f['task'];X=np.random.default_rng(125000+seed).uniform(-np.pi,np.pi,(512,2));y=labels_for(X,task)
  for model,fit in [('fixed',f['baseline']),('learned',f['learned'])]:
   W=np.array(fit.get('input_matrix',np.eye(2)));Y=X@W;freq=fit['frequencies'];bases=['XX','ZZ','XZ','YY'];P=basis_probabilities(Y,freq,bases)
   F,_,covered=measure(P,freq,bases,None,None);truth=elementary_features(Y,freq);error=float(abs(F[:,covered]-truth[:,covered]).max());assert error<1e-12;equivalence.append(error)
   rows.append({'seed':seed,'task':task,'model':model,'budget':0,'repeat':0,**richer_metrics(learned_predict(X,fit),y)})
   for B in (1024,4096):
    for rep in range(10):
     F,used,_=measure(P,freq,bases,B,np.random.default_rng(126000+seed*10000+B+rep));assert used==B
     rows.append({'seed':seed,'task':task,'model':model,'budget':B,'repeat':rep,**richer_metrics(learned_predict(X,fit,F),y)})
 log('38_result',{'rows':rows,'measurement_equivalence_error':max(equivalence),'decision':'Inspect accuracy/loss at finite shots. More optimization or ideal expressivity is insufficient if the readout amplifies sampling errors; distinguish sampling weakness from input-domain generalization.'})

if __name__=='__main__':main()
