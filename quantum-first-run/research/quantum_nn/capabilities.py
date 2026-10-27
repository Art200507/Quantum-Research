"""Capability suite: disjoint data, multiple tasks and strong classical controls."""
from core import *
TASKS=('linear','cos_product','radial','mixed_frequency')

def labels(X,task):
 if task=='linear':f=X[:,0]+.6*X[:,1]
 elif task=='cos_product':f=np.cos(X[:,0])*np.cos(X[:,1])
 elif task=='radial':f=(X**2).sum(1)-2*np.pi
 elif task=='mixed_frequency':f=np.sin(2*X[:,0])+.4*np.cos(3*X[:,1])
 else:raise ValueError(task)
 return (f>0).astype(float)

def splits(seed,task,n):
 X=np.random.default_rng(60000+seed).uniform(-np.pi,np.pi,(768,2));y=labels(X,task)
 return (X[:n],y[:n]),(X[128:256],y[128:256]),(X[256:],y[256:])

def fourier(X):
 # Fixed generic trig basis includes original inputs; no target mask is supplied.
 cols=[X[:,0]/np.pi,X[:,1]/np.pi]
 for a in range(4):
  for b in range(-3,4):
   if a==0 and b<=0:continue
   phase=a*X[:,0]+b*X[:,1];cols.extend([np.cos(phase),np.sin(phase)])
 return np.array(cols).T

def qstates(X,p,layers):
 S=np.zeros((len(X),4),complex);S[:,0]=1
 for layer in range(layers):
  # Data encoding is repeated; trainable rotations are interleaved with data.
  for q in (0,1):
   t=X[:,q]/2;c=np.cos(t);s=np.sin(t)
   pairs=[(0,1),(2,3)] if q==0 else [(0,2),(1,3)]
   old=S.copy()
   for a,b in pairs:S[:,a]=c*old[:,a]-s*old[:,b];S[:,b]=s*old[:,a]+c*old[:,b]
  S=S[:,[0,3,2,1]]
  for q in (0,1):
   S=apply(S,ry(p[4*layer+2*q]),q)
   theta=p[4*layer+2*q+1];U=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)]);S=apply(S,U,q)
 return S

def qcirc(x,p,layers):
 qc=QuantumCircuit(2)
 for layer in range(layers):
  qc.ry(x[0],0);qc.ry(x[1],1);qc.cx(0,1)
  for q in (0,1):qc.ry(p[4*layer+2*q],q);qc.rz(p[4*layer+2*q+1],q)
 return qc

def qfeature(X,p,layers):return abs(qstates(X,p,layers))**2@np.array([1,1,-1,-1])

def fit_model(model,train,val,seed,maxiter=120):
 X,y=train;V,vy=val;r=np.random.default_rng(61000+seed);history=[]
 if model in ('QNN_basic','QNN_reupload'):
  layers=2;theta_count=8 if model=='QNN_reupload' else 2
  feature=lambda A,p:qfeature(A,p,layers) if model=='QNN_reupload' else qnn_z(A,p)
  predict=lambda A,p:p[-2]*feature(A,p[:-2])+p[-1]
  initials=[np.r_[r.normal(0,.3,theta_count),2.,0.] for _ in range(2)]
  objective=lambda p:float(np.mean(np.logaddexp(0,predict(X,p))-y*predict(X,p))+1e-3*np.mean(p**2))
  jac=None
 elif model.startswith('MLP'):
  h=int(model.split('_')[1]);d=2*h+h+h+1
  def predict(A,p):return np.tanh((A/np.pi)@p[:2*h].reshape(2,h)+p[2*h:3*h])@p[3*h:4*h]+p[-1]
  def objective(p):
   A=X/np.pi;W=p[:2*h].reshape(2,h);H=np.tanh(A@W+p[2*h:3*h]);w=p[3*h:4*h];z=H@w+p[-1];dz=(expit(z)-y)/len(y)
   loss=np.mean(np.logaddexp(0,z)-y*z)+1e-3*np.mean(p**2)
   dh=dz[:,None]*w*(1-H**2)
   grad=np.r_[(A.T@dh).ravel(),dh.sum(0),H.T@dz,dz.sum()]+2e-3*p/len(p)
   return loss,grad
  initials=[r.normal(0,.5,d) for _ in range(2)];jac=True
 else:
  transform=(lambda A:A/np.pi) if model=='linear' else fourier
  A=transform(X);predict=lambda A,p:transform(A)@p[:-1]+p[-1]
  def objective(p):
   z=A@p[:-1]+p[-1];dz=(expit(z)-y)/len(y)
   return np.mean(np.logaddexp(0,z)-y*z)+1e-3*np.mean(p**2),np.r_[A.T@dz,dz.sum()]+2e-3*p/len(p)
  initials=[np.zeros(A.shape[1]+1)];jac=True
 best={'validation_loss':float('inf')};runs=[]
 for initial in initials:
  def observe(p):
   z=predict(V,p);loss=float(np.mean(np.logaddexp(0,z)-vy*z))
   if loss<best['validation_loss']:best.update(parameters=p.copy(),validation_loss=loss)
  observe(initial)
  res=minimize(objective,initial,method='L-BFGS-B',jac=jac,callback=observe,options={'maxiter':maxiter,'ftol':1e-8})
  observe(res.x);runs.append({'success':bool(res.success),'message':str(res.message),'iterations':res.nit,'calls':res.nfev})
 best.update(runs=runs,parameter_count=len(best['parameters']))
 return best,predict

def evaluate(z,y):
 p=expit(z);return {'accuracy':float(np.mean((z>0)==y)),'cross_entropy':float(np.mean(np.logaddexp(0,z)-y*z)),'brier':float(np.mean((p-y)**2))}

def main(stage):
 review()
 models=('QNN_basic','linear','MLP_2','MLP_8','fourier') if stage==16 else ('QNN_reupload',)
 log(f'{stage}_plan',{'hypothesis':'Measure learning/generalization across four tasks and sample sizes; success on one quantum-friendly task is not broad capability.' if stage==16 else 'Interleaved data re-uploading may broaden QNN capability; test all tasks, not just the strongest prior result.','models':models,'design':'3 independent data seeds, nested16/64/128 training points;128 disjoint validation and512 test points. Validation selects checkpoints/restarts; test never used for fitting.','tasks':TASKS,'optimizer':'L-BFGS-B120 iterations,2 restarts for QNN/MLP,one convex fit for linear/Fourier; report all failures.','fairness':'Same data. Unequal parameter/function classes and simulator costs reported. Fourier baseline intentionally strong generic nonlinear control. Not a quantum advantage test or full language task.'})
 rows=[];fits=[];checks=[]
 if stage==17:
  r=np.random.default_rng(5);X=r.normal(size=(4,2));p=r.normal(size=8)
  S=qstates(X,p,2)
  for x,s in zip(X,S):checks.append(float(np.max(abs(s-Statevector.from_instruction(qcirc(x,p,2)).data))))
  assert max(checks)<1e-12
 for seed in range(3):
  for task in TASKS:
   for n in (16,64,128):
    train,val,test=splits(seed,task,n)
    for model in models:
     start=time.perf_counter();fit,predict=fit_model(model,train,val,seed);p=fit['parameters']
     fits.append({'seed':seed,'task':task,'n_train':n,'model':model,'seconds':time.perf_counter()-start,**fit})
     for split,ds in [('train',train),('validation',val),('test',test)]:
      X,y=ds;rows.append({'seed':seed,'task':task,'n_train':n,'model':model,'split':split,**evaluate(predict(X,p),y)})
 log(f'{stage}_result',{'rows':rows,'fits':fits,'circuit_errors':checks,'decision':'Review task-specific learning curves, generalization gaps and optimizer failures before choosing reliability interventions. Aggregate task accuracy alone can hide failure modes.'})

if __name__=='__main__':
 import sys
 main(int(sys.argv[1]))
