"""Experiment20: retain shallow features in a learned two-circuit QNN mixture."""
from capabilities import *
ACTIVE=np.array([0,1,2,3,6])

def branch_features(X,p):
 theta=np.zeros(8);theta[ACTIVE]=p[1:6]
 return qnn_z(X,[0,p[0]]),qfeature(X,theta,2)

def mixture_logit(X,p):
 a,b=branch_features(X,p);weight=expit(p[6])
 return p[7]*(weight*a+(1-weight)*b)+p[8]

def main():
 review();previous=json.loads((LOG/'19_result.json').read_text())
 log('20_plan',{'hypothesis':'Retaining a shallow branch may preserve earlier useful features while allowing learned re-uploaded features for other tasks.','architecture':'Two separately measured two-qubit circuits, classical sigmoid-weighted combination, classical gain/bias. Remove unobservable rotations:9 effective trainable parameters, same count as MLP_2. Not a single deeper circuit or unique quantum operation.','design':'Same3 data seeds and4 tasks,16/64/128 training points, independent validation/test. Two restarts,120 optimization iterations, train all9 parameters. Compare saved baselines and independently fit identical-data MLP_2 already in16.','cost':'Two circuit settings per prediction versus one; total inference shots must be split between branches. Training uses ideal simulated expectations.','reviewed_structural_tests':len(previous['rows'])})
 rows=[];fits=[];identity=[]
 # Verify dropping unobservable angles leaves deep branch unchanged.
 r=np.random.default_rng(67000);X=r.normal(size=(64,2));theta=r.normal(size=8);pruned=theta.copy();pruned[[4,5,7]]=0
 identity.append(float(np.max(abs(qfeature(X,theta,2)-qfeature(X,pruned,2)))));assert max(identity)<1e-12
 for seed in range(3):
  for task in TASKS:
   for n in (16,64,128):
    train,val,test=splits(seed,task,n);r=np.random.default_rng(68000+seed);best={'loss':float('inf')};runs=[]
    def objective(p):
     z=mixture_logit(train[0],p);return np.mean(np.logaddexp(0,z)-train[1]*z)+1e-3*np.mean(p**2)
    def observe(p):
     z=mixture_logit(val[0],p);loss=float(np.mean(np.logaddexp(0,z)-val[1]*z))
     if loss<best['loss']:best.update(loss=loss,parameters=p.copy())
    for _ in range(2):
     initial=np.r_[r.normal(0,.3,6),0.,2.,0.];observe(initial)
     res=minimize(objective,initial,method='L-BFGS-B',callback=observe,options={'maxiter':120,'ftol':1e-8});observe(res.x)
     runs.append({'success':bool(res.success),'message':str(res.message),'iterations':res.nit,'calls':res.nfev})
    p=best['parameters'];fits.append({'seed':seed,'task':task,'n_train':n,'parameters':p,'validation_loss':best['loss'],'runs':runs,'shallow_weight':float(expit(p[6]))})
    for split,(X,y) in [('train',train),('validation',val),('test',test)]:rows.append({'seed':seed,'task':task,'n_train':n,'model':'QNN_mixture','split':split,**evaluate(mixture_logit(X,p),y)})
 log('20_result',{'rows':rows,'fits':fits,'pruned_rotation_errors':identity,'decision':'Review performance task by task and branch weights. Mixture success is not automatically quantum-specific; test finite shots and calibration before calling it reliable.'})

if __name__=='__main__':main()
