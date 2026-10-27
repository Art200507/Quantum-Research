"""Experiment22: held-out input-region generalization, rather than IID only."""
from qnn_mixture import *

def fit_mix(train,val,seed):
 r=np.random.default_rng(73000+seed);best={'loss':float('inf')};runs=[]
 def objective(p):
  z=mixture_logit(train[0],p);return np.mean(np.logaddexp(0,z)-train[1]*z)+1e-3*np.mean(p**2)
 def observe(p):
  z=mixture_logit(val[0],p);loss=float(np.mean(np.logaddexp(0,z)-val[1]*z))
  if loss<best['loss']:best.update(loss=loss,parameters=p.copy())
 for _ in range(2):
  init=np.r_[r.normal(0,.3,6),0.,2.,0.];observe(init)
  res=minimize(objective,init,method='L-BFGS-B',callback=observe,options={'maxiter':120,'ftol':1e-8});observe(res.x)
  runs.append({'success':bool(res.success),'message':str(res.message),'iterations':res.nit})
 return {'parameters':best['parameters'],'validation_loss':best['loss'],'runs':runs},mixture_logit

def fit_feature(train,val):
 phi=lambda X:np.c_[np.cos(X[:,0])*np.cos(X[:,1]),np.sin(X[:,1])]
 X,y=train;A=phi(X)
 def objective(p):
  z=A@p[:2]+p[2];dz=(expit(z)-y)/len(y)
  return np.mean(np.logaddexp(0,z)-y*z)+1e-3*np.mean(p**2),np.r_[A.T@dz,dz.sum()]+2e-3*p/3
 fit=minimize(objective,np.zeros(3),jac=True,method='L-BFGS-B',options={'maxiter':120})
 return {'parameters':fit.x,'success':bool(fit.success)},lambda X,p:phi(X)@p[:2]+p[2]

def main():
 review();log('22_plan',{'hypothesis':'Random-split accuracy can hide failure on unseen combinations. Test an entire held-out positive-positive quadrant, while validation remains in the training-support region.','design':'3 new data seeds;128 train/128 validation from other3 quadrants,512 seen-region test and512 held-out-region test. Cos-product/radial/mixed-frequency tasks only; linear target omitted because held-out quadrant has only one class.','models':'9-parameter quantum mixture,9-parameter MLP,33-parameter MLP,3-parameter encoding-matched classical feature control. Two restarts120 iterations where applicable.','fairness':'No labels from held-out quadrant are used in training, validation, checkpoint selection or temperature tuning. Architecture-informed control is explicit. No finite-shot inference in this test.'})
 rows=[];fits=[]
 for seed in range(3):
  pool=np.random.default_rng(71000+seed).uniform(-np.pi,np.pi,(4096,2));seen=pool[~((pool[:,0]>0)&(pool[:,1]>0))]
  pool=np.random.default_rng(72000+seed).uniform(-np.pi,np.pi,(4096,2));held=pool[(pool[:,0]>0)&(pool[:,1]>0)][:512];assert len(held)==512
  trainX=seen[:128];valX=seen[128:256];testX=seen[256:768]
  assert np.all((held>0).all(1)) and not np.any((trainX>0).all(1))
  for task in ('cos_product','radial','mixed_frequency'):
   train=(trainX,labels(trainX,task));val=(valX,labels(valX,task))
   for model in ('QNN_mixture','MLP_2','MLP_8','feature_control'):
    fit,predict=fit_mix(train,val,seed) if model=='QNN_mixture' else fit_feature(train,val) if model=='feature_control' else fit_model(model,train,val,seed)
    fits.append({'seed':seed,'task':task,'model':model,**fit})
    for split,X in [('train',trainX),('validation',valX),('seen_test',testX),('held_out_quadrant',held)]:
     y=labels(X,task);z=predict(X,fit['parameters']);rows.append({'seed':seed,'task':task,'model':model,'split':split,'positive_label_fraction':float(y.mean()),**evaluate(z,y)})
 log('22_result',{'rows':rows,'fits':fits,'decision':'Contrast seen-region and held-out-region accuracy; a gap is evidence about this distribution shift, not a universal capacity limitation. Feature-matched success would undermine a uniquely quantum generalization claim.'})

if __name__=='__main__':main()
