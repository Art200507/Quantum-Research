"""Experiment29: choose model using validation only, assess fresh test data."""
from measured_features import *

def classical_predict(X,fit):
 model=fit['model'];p=np.array(fit['parameters'])
 if model=='fourier':return fourier(X)@p[:-1]+p[-1]
 h=int(model.split('_')[1]);return np.tanh((X/np.pi)@p[:2*h].reshape(2,h)+p[2*h:3*h])@p[3*h:4*h]+p[-1]

def main():
 review();log('29_plan',{'hypothesis':'Validation-based selection may retain single-scale strengths while using multi-scale features for tasks that need them. Test selection on fresh points, not a per-test oracle.','design':'Use only saved validation cross-entropy to choose one quantum model among25/26 and one classical control among MLP2/MLP8/Fourier. Fresh512 test points per seed/task;5 seeds,4 tasks.','measurements':'For chosen quantum head, ideal and total1024 shots over4 bases/view,10 repeats. All setting costs charged.','limits':'Per-task supervised model selection, not one jointly trained universal or task-agnostic QNN. Validation ideal, not finite-shot optimized. Does not establish quantum superiority.'})
 fits25=json.loads((LOG/'25_result.json').read_text())['fits'];fits26=json.loads((LOG/'26_result.json').read_text())['fits'];allfits=fits25+fits26;selected=[];rows=[];bases=['XX','ZZ','XZ','YY']
 for seed in range(5):
  X=np.random.default_rng(90000+seed).uniform(-np.pi,np.pi,(512,2))
  for task in TASKS:
   q=[f for f in allfits if f['seed']==seed and f['task']==task and f['model']=='quantum_features'];best=min(q,key=lambda f:f['validation_loss'])
   c=[f for f in allfits if f['seed']==seed and f['task']==task and f['model'] in ('MLP_2','MLP_8','fourier')];bc=min(c,key=lambda f:f['validation_loss'])
   selected.append({'seed':seed,'task':task,'quantum_fit':best,'classical_fit':bc})
   y=labels(X,task);rows.append({'seed':seed,'task':task,'model':'selected_quantum','budget':0,'repeat':0,**metrics(readout(X,best),y)})
   rows.append({'seed':seed,'task':task,'model':'selected_classical','budget':0,'repeat':0,**metrics(classical_predict(X,bc),y)})
   probs=basis_probabilities(X,best['frequencies'],bases)
   for rep in range(10):
    F,used,cov=measure(probs,best['frequencies'],bases,1024,np.random.default_rng(90500+seed*1000+rep));assert used==1024
    rows.append({'seed':seed,'task':task,'model':'selected_quantum','budget':used,'repeat':rep,**metrics(readout(X,best,F),y)})
 log('29_result',{'rows':rows,'selected':selected,'decision':'Inspect fresh-test task breadth and compare to validation-selected classical models. Use disjoint calibration data next; do not reuse model-selection validation for uncertainty guarantees.'})

if __name__=='__main__':main()
