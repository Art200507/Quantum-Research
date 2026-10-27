"""Sequential QNN and quantum-attention experiments. Read logs before each run."""
import argparse,json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,softmax
from core import *

def qnn_experiment():
 review();log('01_plan',{'hypothesis':'A trained two-qubit QNN learns nonlinear angular parity; compare raw linear and encoding-matched classical features on disjoint inputs.','bounds':'5 seeds, 128 train/128 validation/512 test points, ideal local simulation; max80 QNN iterations.'})
 rows=[];fits=[]
 for seed in range(5):
  r=np.random.default_rng(3000+seed);X=r.uniform(-np.pi,np.pi,(768,2));y=(np.cos(X[:,0])*np.cos(X[:,1])>0).astype(int)
  train=slice(0,128);valid=slice(128,256);test=slice(256,None)
  # Two trainable RY parameters plus trainable classical gain and bias.
  def loss(p):
   z=p[2]*qnn_z(X[train],p[:2])+p[3]
   return np.mean(np.logaddexp(0,z)-y[train]*z)+1e-4*p[2]**2
  init=np.r_[r.normal(0,.2,2),2.,0.];res=minimize(loss,init,method='L-BFGS-B',options={'maxiter':80})
  p=res.x;fits.append({'seed':seed,'parameters':p,'success':bool(res.success),'message':str(res.message),'iterations':res.nit,'function_calls':res.nfev})
  phi=lambda A:np.stack([np.cos(A[:,0])*np.cos(A[:,1]),np.sin(A[:,0])*np.cos(A[:,1]),np.sin(A[:,1])],1)
  for name,features in [('raw_linear',lambda A:A),('matched_classical',phi)]:
   tr=features(X[train]);fit=fit_logistic(tr,y[train]);z=features(X[test])@fit.x[:-1]+fit.x[-1]
   rows.append({'seed':seed,'model':name,'test_accuracy':float(np.mean((z>0)==y[test]))})
  z=qnn_z(X[test],p[:2]);rows.append({'seed':seed,'model':'trained_QNN_ideal','test_accuracy':float(np.mean((p[2]*z+p[3]>0)==y[test])),'validation_accuracy':float(np.mean((p[2]*qnn_z(X[valid],p[:2])+p[3]>0)==y[valid]))})
  for shots in (32,128,1024):
   estimates=2*r.binomial(shots,(1+z)/2)/shots-1
   rows.append({'seed':seed,'model':f'trained_QNN_{shots}_shots','test_accuracy':float(np.mean((p[2]*estimates+p[3]>0)==y[test]))})
 log('01_result',{'validation_statevector_max_error':check(),'rows':rows,'fits':fits,'decision':'Review mean accuracy against matched-classical and finite-shot costs before moving to quantum attention. This task is constructed to match the encoding, not a natural benchmark or advantage test.'})

if __name__=='__main__':qnn_experiment()
