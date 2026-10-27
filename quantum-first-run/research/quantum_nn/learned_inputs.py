"""Train input directions/scales jointly with quantum-observable readout.

Circuit: RY(f*(x@W)[0]), RY(f*(x@W)[1]), CX; measured Pauli outputs.
Analytic two-qubit simulator/gradient is independently checked against Qiskit.
"""
from feature_control import *
from new_tasks import new_labels,richer_metrics

def feature_derivatives(X,frequencies):
 blocks=[]
 for f in frequencies:
  a,b=(X*f).T;sa,ca,sb,cb=np.sin(a),np.cos(a),np.sin(b),np.cos(b);zero=np.zeros(len(X))
  da={'IX':ca*sb,'IZ':-sa,'XX':ca,'XZ':-sa*sb,'YY':-ca*cb,'ZI':-sa*cb}
  db={'IX':sa*cb,'XI':cb,'XZ':ca*cb,'YY':sa*sb,'ZI':-ca*sb,'ZZ':-sb}
  blocks.append(np.stack([np.column_stack([da.get(p,zero) for p in PAULIS]),np.column_stack([db.get(p,zero) for p in PAULIS])],axis=2)*f)
 return np.concatenate(blocks,axis=1)

def joint_objective(p,X,y,base):
 W=p[:4].reshape(2,2);weights=p[4:];Y=X@W;mask=np.array(base['mask'],bool);scale=np.array(base['scale']);mean=np.array(base['mean'])
 F=elementary_features(Y,base['frequencies'])[:,mask];A=(F-mean)/scale;z=A@weights[:-1]+weights[-1];dz=(expit(z)-y)/len(y)
 lam=base['lambda_'];loss=np.mean(np.logaddexp(0,z)-y*z)+lam*np.mean(weights**2)+.001*np.sum((W-np.eye(2))**2)
 dF=feature_derivatives(Y,base['frequencies'])[:,mask,:];dY=np.einsum('nkj,k->nj',dF,weights[:-1]/scale)
 gradW=X.T@(dz[:,None]*dY)+.002*(W-np.eye(2));gradw=np.r_[A.T@dz,dz.sum()]+2*lam*weights/len(weights)
 return float(loss),np.r_[gradW.ravel(),gradw]

def learned_predict(X,fit,F=None):
 W=np.array(fit.get('input_matrix',np.eye(2)));return readout(X@W,fit,F)

def joint_fit(train,val,base,seed):
 X,y=train;V,vy=val;r=np.random.default_rng(121000+seed);best={'validation_loss':base['validation_loss'],'input_matrix':np.eye(2),'weights':np.array(base['weights'])};runs=[]
 def observe(p):
  candidate={**base,'input_matrix':p[:4].reshape(2,2),'weights':p[4:]};z=learned_predict(V,candidate);loss=float(np.mean(np.logaddexp(0,z)-vy*z))
  if loss<best['validation_loss']:best.update(validation_loss=loss,input_matrix=p[:4].reshape(2,2).copy(),weights=p[4:].copy())
 for start in range(2):
  W=np.eye(2) if start==0 else np.eye(2)+r.normal(0,.2,(2,2));initial=np.r_[W.ravel(),np.array(base['weights'])];observe(initial)
  res=minimize(joint_objective,initial,args=(X,y,base),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*4+[(None,None)]*len(base['weights']),callback=observe,options={'maxiter':200,'ftol':1e-9})
  observe(res.x);runs.append({'success':bool(res.success),'message':str(res.message),'iterations':res.nit,'calls':res.nfev})
 return {**base,**best,'encoder_runs':runs,'learned_parameter_count':base['readout_parameters']+4}

def labels_for(X,task):return new_labels(X,task) if task in ('rotated_ellipse','diagonal_wave','annulus','tilted_interaction') else labels(X,task)

def derivative_check():
 r=np.random.default_rng(122000);X=r.normal(size=(20,2));y=r.integers(2,size=20);base=fit_readout((X,y),(X,y),[1.]);p=np.r_[(np.eye(2)+.1*r.normal(size=(2,2))).ravel(),r.normal(size=len(base['weights']))]
 loss,g=joint_objective(p,X,y,base);numerical=[]
 for j in range(len(p)):
  d=np.zeros(len(p));d[j]=1e-5;numerical.append((joint_objective(p+d,X,y,base)[0]-joint_objective(p-d,X,y,base)[0])/2e-5)
 err=float(np.max(abs(g-numerical)));assert err<1e-7;return err

def main():
 review();log('35_plan',{'hypothesis':'Learning four input-mixing coefficients may fix the diagonal-wave failure while retaining richer quantum outputs on other tasks.','tasks':['diagonal_wave','tilted_interaction','cos_product','radial'],'design':'5 fresh data seeds,128 train/256 validation/1024 test, single or4-frequency views. Retrain frozen-encoder baseline, jointly learn W+readout from two starts/config,select checkpoints/configuration by validation only.','optimization':'Analytic simulator gradient checked by finite differences;200 L-BFGS-B iterations. Readout regularization from baseline validation; W regularized toward identity. All convergence outcomes logged.','scope':'Hybrid trainable encoder: data-dependent gate angles f*(x@W). Additional4 parameters and extra optimization; not a matched-training-cost comparison. No quantum-only expressivity claim.'})
 error=derivative_check();rows=[];fits=[];circuit_errors=[]
 for seed in range(5):
  X=np.random.default_rng(120000+seed).uniform(-np.pi,np.pi,(1408,2))
  for task in ('diagonal_wave','tilted_interaction','cos_product','radial'):
   y=labels_for(X,task);train=(X[:128],y[:128]);val=(X[128:384],y[128:384]);Xt,yt=X[384:],y[384:]
   baseline=[fit_readout(train,val,f) for f in ([1.],[.5,1.,2.,3.])];learned=[joint_fit(train,val,b,seed) for b in baseline]
   b=min(baseline,key=lambda f:f['validation_loss']);l=min(learned,key=lambda f:f['validation_loss']);fits.append({'seed':seed,'task':task,'baseline':b,'learned':l,'all_learned_candidates':learned})
   for model,fit in [('fixed_inputs',b),('learned_inputs',l)]:rows.append({'seed':seed,'task':task,'model':model,**richer_metrics(learned_predict(Xt,fit),yt)})
   Y=Xt[:2]@l['input_matrix'];F=elementary_features(Y,l['frequencies'])
   for i,point in enumerate(Y):
    for j,f in enumerate(l['frequencies']):
     S=Statevector.from_instruction(map_circuit(point,f));actual=np.array([S.expectation_value(Pauli(p)).real for p in PAULIS]);circuit_errors.append(float(abs(F[i,15*j:15*(j+1)]-actual).max()))
   print('Completed',seed,task,flush=True)
 assert max(circuit_errors)<1e-12
 log('35_result',{'rows':rows,'fits':fits,'gradient_error':error,'trained_circuit_errors':circuit_errors,'decision':'Review learning gains and regressions; freeze encoder-training protocol before a confirmation batch with different wave directions and new cohorts. Then test physical measurement budgets.'})

if __name__=='__main__':main()
