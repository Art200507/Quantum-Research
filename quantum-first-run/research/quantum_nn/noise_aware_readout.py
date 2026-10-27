"""42: refit a frozen learned encoder's readout using measured features."""
from nonwave_confirmation import *
from measured_features import basis_probabilities,measure
BASES=['XX','ZZ','XZ','YY']
def noisy_features(X,fit,budget,r):
 P=basis_probabilities(X@np.array(fit['input_matrix']),fit['frequencies'],BASES)
 F,used,covered=measure(P,fit['frequencies'],BASES,budget,r)
 assert used==budget and np.all(covered[np.array(fit['mask'],bool)])
 return F

def refit(F,y,V,vy,base):
 mask=np.array(base['mask'],bool);A=(F[:,mask]-base['mean'])/base['scale'];C=(V[:,mask]-base['mean'])/base['scale'];best=None;runs=[]
 for lam in (.001,.01,.1,1.):
  def obj(w):
   z=A@w[:-1]+w[-1];d=(expit(z)-y)/len(y)
   return np.mean(np.logaddexp(0,z)-y*z)+lam*np.mean(w*w),np.r_[A.T@d,d.sum()]+2*lam*w/len(w)
  res=minimize(obj,np.array(base['weights']),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
  z=C@res.x[:-1]+res.x[-1];loss=float(np.mean(np.logaddexp(0,z)-vy*z));runs.append(dict(success=bool(res.success),iterations=res.nit,lambda_=lam,validation_loss=loss))
  if best is None or loss<best['validation_loss']:best={**base,'weights':res.x,'lambda_':lam,'validation_loss':loss}
 return {**best,'noise_readout_runs':runs}
def main():
 review();previous=json.loads((LOG/'41_result.json').read_text())
 log('42_plan',{'hypothesis':'Measured-feature readout training may reduce shot noise failures on non-wave tasks without changing the encoder.','reviewed':'41_result; all four non-wave tasks retained, no selection by best test result.','design':'Frozen stage41 learned encoders; reuse original128train/256validation points with independent measurement draws. Four augmentation draws per point at256 or1024 total shots each; lambda selected on augmented validation CE. Fresh1024test points,5 paired measurement repeats. Same measured test features for ideal-trained and noise-trained readouts.','cost':'Per model/budget:512*budget training shots and1024*budget validation shots, additional to earlier ideal encoder training. Test5120*budget shots. Augmentations are repeated observations, not independent labeled examples; encoder remains ideal-trained.','scope':'Sampling noise only; no gate noise,hardware,uncertainty-coverage or quantum advantage claims.'})
 rows=[];fits=[];errors=[]
 for item in previous['fits']:
  seed,task,base=item['seed'],item['task'],item['learned'];X=np.random.default_rng(131000+seed).uniform(-np.pi,np.pi,(1408,2));y=target41(X,task)
  T=np.random.default_rng(132000+seed).uniform(-np.pi,np.pi,(1024,2));ty=target41(T,task)
  for B in (256,1024):
   r=np.random.default_rng(133000+seed*10000+B+TASKS41.index(task)*100)
   F=np.concatenate([noisy_features(X[:128],base,B,r) for _ in range(4)])
   V=np.concatenate([noisy_features(X[128:384],base,B,r) for _ in range(4)])
   fit=refit(F,np.tile(y[:128],4),V,np.tile(y[128:384],4),base);fits.append(dict(seed=seed,task=task,budget=B,fit=fit))
   for rep in range(5):
    Ft=noisy_features(T,base,B,r)
    for name,f in [('ideal_training',base),('noise_training',fit)]:rows.append(dict(seed=seed,task=task,budget=B,repeat=rep,model=name,**richer_metrics(learned_predict(T,f,Ft),ty)))
  print(seed,task,flush=True)
 log('42_result',{'rows':rows,'fits':fits,'decision':'Review paired accuracy and CE by task and seed. Noise training is a reliability option only where validated, not a universal fix; next independent confirmation or gate-noise test must be motivated by these results.'})
if __name__=='__main__':main()
