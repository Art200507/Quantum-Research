"""Experiment37: continue selected encoder fits and expose optimization traces."""
from learned_inputs import *

def main():
 review();log('37_plan',{'hypothesis':'Many encoder runs hit the iteration cap; continuing them may improve training without improving held-out predictions. Distinguish convergence from capability.','design':'Continue selected35 checkpoints for diagonal_wave/cos_product,5 seeds,400 additional iterations,unchanged objective/bounds. Original128 train/256 validation retained; test on an entirely new1024-point cohort.','selection':'Keep previous checkpoint unless validation loss improves; never select on new test. Log training/validation per iteration and final optimizer status.','limits':'Targeted convergence diagnostic, not matched optimization budget or new algorithm. Smaller loss is not automatically better generalization.'})
 old=json.loads((LOG/'35_result.json').read_text())['fits'];rows=[];fits=[]
 for f in old:
  if f['task'] not in ('diagonal_wave','cos_product'):continue
  seed,task=f['seed'],f['task'];base=f['learned'];X=np.random.default_rng(120000+seed).uniform(-np.pi,np.pi,(1408,2));y=labels_for(X,task);train=(X[:128],y[:128]);val=(X[128:384],y[128:384])
  Xt=np.random.default_rng(124000+seed).uniform(-np.pi,np.pi,(1024,2));yt=labels_for(Xt,task);initial=np.r_[np.array(base['input_matrix']).ravel(),base['weights']]
  best={'validation_loss':base['validation_loss'],'parameters':initial.copy()};trace=[]
  def observe(p):
   fit={**base,'input_matrix':p[:4].reshape(2,2),'weights':p[4:]};z=learned_predict(val[0],fit);vl=float(np.mean(np.logaddexp(0,z)-val[1]*z));tl=float(joint_objective(p,*train,base)[0]);trace.append({'iteration':len(trace),'training_objective':tl,'validation_loss':vl})
   if vl<best['validation_loss']:best.update(validation_loss=vl,parameters=p.copy())
  observe(initial);res=minimize(joint_objective,initial,args=(*train,base),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*4+[(None,None)]*len(base['weights']),callback=observe,options={'maxiter':400,'ftol':1e-9});observe(res.x)
  p=best['parameters'];refined={**base,'input_matrix':p[:4].reshape(2,2),'weights':p[4:],'validation_loss':best['validation_loss']}
  fits.append({'seed':seed,'task':task,'before':base,'refined':refined,'optimizer_success':bool(res.success),'message':str(res.message),'iterations':res.nit,'trace':trace})
  for model,fit in [('before',base),('validation_selected_refinement',refined)]:rows.append({'seed':seed,'task':task,'model':model,**richer_metrics(learned_predict(Xt,fit),yt)})
 log('37_result',{'rows':rows,'fits':fits,'decision':'Inspect whether continued optimization helps new data; retain baseline checkpoints when validation favors them. Use frozen model choices for measurement-budget tests.'})

if __name__=='__main__':main()
