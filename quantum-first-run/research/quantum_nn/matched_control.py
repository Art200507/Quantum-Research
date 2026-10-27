"""Experiment18: audit apparent capability gains with an exact function-class control."""
from capabilities import *

def main():
 review();log('18_plan',{'hypothesis':'The basic QNN cosine-task advantage may be entirely due to its fixed feature family. Test an exact classical representation, not only a generic NN trained on raw features.','derivation':'Z1 expectation = cos(theta1)*cos(x0)*cos(x1) - sin(theta1)*sin(x1). theta0 has no effect. Classical logit A*cos(x0)*cos(x1)+B*sin(x1)+bias reproduces any fitted basic QNN.','design':'Verify identity for every saved basic-QNN fit on independent points; train3-parameter convex feature control on identical splits across all4 tasks and3 sample sizes.','fairness':'Architecture-derived features, no additional labels. This is a diagnostic control and does not show all larger QNNs can be emulated efficiently.'})
 rows=[];fits=[];errors=[]
 old=json.loads((LOG/'16_result.json').read_text())
 phi=lambda X:np.c_[np.cos(X[:,0])*np.cos(X[:,1]),np.sin(X[:,1])]
 X=np.random.default_rng(65000).uniform(-np.pi,np.pi,(512,2))
 for f in old['fits']:
  if f['model']!='QNN_basic':continue
  p=np.array(f['parameters']);coef=np.array([p[-2]*np.cos(p[1]),-p[-2]*np.sin(p[1])]);z=phi(X)@coef+p[-1]
  err=float(np.max(abs(z-(p[-2]*qnn_z(X,p[:2])+p[-1]))));assert err<1e-10;errors.append(err)
 for seed in range(3):
  for task in TASKS:
   for n in (16,64,128):
    train,val,test=splits(seed,task,n);A=phi(train[0]);y=train[1]
    def objective(p):
     z=A@p[:2]+p[2];dz=(expit(z)-y)/len(y)
     return np.mean(np.logaddexp(0,z)-y*z)+1e-3*np.mean(p**2),np.r_[A.T@dz,dz.sum()]+2e-3*p/3
    res=minimize(objective,np.zeros(3),jac=True,method='L-BFGS-B',options={'maxiter':120})
    fits.append({'seed':seed,'task':task,'n_train':n,'parameters':res.x,'success':bool(res.success),'iterations':res.nit})
    for split,(X,y) in [('train',train),('validation',val),('test',test)]:rows.append({'seed':seed,'task':task,'n_train':n,'model':'exact_feature_classical','split':split,**evaluate(phi(X)@res.x[:2]+res.x[2],y)})
 log('18_result',{'rows':rows,'fits':fits,'maximum_QNN_classical_logit_error':max(errors),'decision':'Any advantage over generic baselines in this basic QNN is not unique quantum capability. Next investigate why re-uploading broadens some tasks while losing the cosine interaction.'})

if __name__=='__main__':main()
