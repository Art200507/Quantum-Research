"""Experiment39: fixed learned encoders outside their training input range."""
from learned_measurements import *

def main():
 review();log('39_plan',{'hypothesis':'Learning input directions may improve in-range capability yet still fail outside the training range.','design':'Freeze models/checkpoints35/37.5 seeds,1024 fresh inputs per seed,range multipliers1/1.5/2/3. Recompute labels from the unchanged analytic task at the new input; this is domain extrapolation, not noisy observation of the old input.','metrics':'Accuracy and balanced accuracy plus majority baseline, since radial labels become imbalanced when the domain grows. Exact expectations isolate representation error; no finite-shot or gate noise here.','limits':'Chosen analytic tasks,not natural real-world data. Shared base points across scales are paired,not independent. No model selection or fitting on extrapolation labels.'})
 rows=[];control_errors=[]
 for f in frozen_models():
  seed,task=f['seed'],f['task'];base=np.random.default_rng(127000+seed).uniform(-np.pi,np.pi,(1024,2))
  for scale in (1.,1.5,2.,3.):
   X=base*scale;y=labels_for(X,task)
   for model,fit in [('fixed',f['baseline']),('learned',f['learned'])]:
    z=learned_predict(X,fit);rows.append({'seed':seed,'task':task,'scale':scale,'model':model,**richer_metrics(z,y)})
    W=np.array(fit.get('input_matrix',np.eye(2)));quantum=readout(X@W,fit,features(X@W,fit['frequencies']));control_errors.append(float(np.max(abs(z-quantum))))
 assert max(control_errors)<1e-9
 log('39_result',{'rows':rows,'classical_formula_vs_quantum_feature_logit_error':max(control_errors),'decision':'Interpret balanced accuracy under extrapolation separately from in-range measurement gains. Do not infer universal generalization or uniquely quantum ability from periodic toy tasks.'})

if __name__=='__main__':main()
