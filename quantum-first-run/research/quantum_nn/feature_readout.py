"""Quantum feature readout: retain multiple observables instead of one scalar."""
from capabilities import *
from compositional import fit_mix
from qiskit.quantum_info import Pauli
PAULIS=[a+b for a in 'IXYZ' for b in 'IXYZ' if a+b!='II']
OPS=np.array([Pauli(s).to_matrix() for s in PAULIS])

def map_states(X,f):return encoded(X*f)[:,[0,3,2,1]]

def map_circuit(x,f):
 qc=QuantumCircuit(2);qc.ry(f*x[0],0);qc.ry(f*x[1],1);qc.cx(0,1);return qc

def features(X,frequencies):
 blocks=[]
 for f in frequencies:
  S=map_states(X,f);blocks.append(np.einsum('bi,kij,bj->bk',S.conj(),OPS,S).real)
 return np.concatenate(blocks,1)

def fresh_splits(seed,task):
 X=np.random.default_rng(80000+seed).uniform(-np.pi,np.pi,(1408,2));y=labels(X,task)
 return (X[:128],y[:128]),(X[128:384],y[128:384]),(X[384:],y[384:])

def fit_readout(train,val,frequencies):
 X,y=train;F=features(X,frequencies);mask=F.std(0)>1e-8
 mu=F[:,mask].mean(0);scale=F[:,mask].std(0);A=(F[:,mask]-mu)/scale
 V=(features(val[0],frequencies)[:,mask]-mu)/scale;vy=val[1];best={'validation_loss':float('inf')};runs=[]
 for lam in (.001,.01,.1,1.):
  def objective(p):
   z=A@p[:-1]+p[-1];dz=(expit(z)-y)/len(y)
   return np.mean(np.logaddexp(0,z)-y*z)+lam*np.mean(p**2),np.r_[A.T@dz,dz.sum()]+2*lam*p/len(p)
  res=minimize(objective,np.zeros(A.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
  z=V@res.x[:-1]+res.x[-1];loss=float(np.mean(np.logaddexp(0,z)-vy*z));runs.append({'lambda':lam,'success':bool(res.success),'iterations':res.nit,'validation_loss':loss})
  if loss<best['validation_loss']:best.update(validation_loss=loss,weights=res.x.copy(),lambda_=lam)
 best.update(frequencies=list(frequencies),mask=mask,mean=mu,scale=scale,readout_parameters=A.shape[1]+1,runs=runs)
 return best

def readout(X,fit,F=None):
 if F is None:F=features(X,fit['frequencies'])
 mask=np.array(fit['mask'],bool);w=np.array(fit['weights'])
 return ((F[:,mask]-fit['mean'])/fit['scale'])@w[:-1]+w[-1]

def checks():
 r=np.random.default_rng(81000);X=r.uniform(-np.pi,np.pi,(4,2));error=0
 for f in (.5,1.,2.,3.):
  F=features(X,[f])
  for i,x in enumerate(X):
   S=Statevector.from_instruction(map_circuit(x,f));actual=np.array([S.expectation_value(Pauli(p)).real for p in PAULIS]);error=max(error,float(abs(F[i]-actual).max()))
 assert error<1e-12
 return error

def main(stage):
 review();frequencies=[1.] if stage==25 else [.5,1.,2.,3.]
 log(f'{stage}_plan',{'hypothesis':'Our readout may discard useful quantum features: compare multiple Pauli outputs with the prior single-scalar mixture.' if stage==25 else 'A frequency bank may retain useful features across tasks instead of losing them through deeper encoding.','architecture':'Fixed two-qubit RY encoding+CX,measure all15 nonidentity Pauli observables,train classical readout. Hybrid quantum feature model, not end-to-end trained quantum gates.','frequencies':frequencies,'design':'5 fresh seeds,4 tasks,128 training/256 validation/1024 test. Regularization selected only on validation. Known-zero feature coordinates removed using training variance.','measurement_cost':'Up to9 local Pauli measurement settings per frequency, shared shots within settings; ideal expectations here. Additional outputs are not free.','controls':'Retrain9-parameter QNN mixture and9-parameter MLP on same data in25. Multi-frequency model in26 has more features, so compare to matched-count MLP8 and generic Fourier as well.','novelty':'Multiple observables/encoding frequencies are established quantum-feature ideas, not a novelty claim.'})
 rows=[];fits=[];err=checks()
 for seed in range(5):
  for task in TASKS:
   train,val,test=fresh_splits(seed,task)
   fit=fit_readout(train,val,frequencies);fits.append({'seed':seed,'task':task,'model':'quantum_features',**fit})
   for split,(X,y) in [('train',train),('validation',val),('test',test)]:rows.append({'seed':seed,'task':task,'model':'quantum_features','split':split,**evaluate(readout(X,fit),y)})
   control_models=['QNN_mixture','MLP_2'] if stage==25 else ['MLP_8','fourier']
   for model in control_models:
    control,predict=fit_mix(train,val,seed) if model=='QNN_mixture' else fit_model(model,train,val,seed)
    fits.append({'seed':seed,'task':task,'model':model,**control})
    for split,(X,y) in [('train',train),('validation',val),('test',test)]:rows.append({'seed':seed,'task':task,'model':model,'split':split,**evaluate(predict(X,control['parameters']),y)})
 log(f'{stage}_result',{'rows':rows,'fits':fits,'qiskit_feature_error':err,'decision':'Review breadth and matched-data controls, then test actual measurement budgets before claiming reliability or a useful quantum improvement.'})

if __name__=='__main__':
 import sys
 main(int(sys.argv[1]))
