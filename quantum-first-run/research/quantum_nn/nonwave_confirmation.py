"""41: frozen learned-encoder protocol on non-wave decision boundaries."""
from learned_inputs import *
TASKS41=('rotated_ellipse','annulus','parabola','xor_quadrants')
def target41(X,task):
 if task in ('rotated_ellipse','annulus'):return new_labels(X,task)
 a,b=X.T
 return ((b>.5*a*a-1.5) if task=='parabola' else (a*b>0)).astype(float)
def main():
 review();log('41_plan',{'hypothesis':'Encoder learning may improve non-wave boundaries, or reveal specialization.','tasks':TASKS41,'design':'3 fresh seeds 128 train/256 validation/1024 test; frozen stage35 protocol; fixed/learned single or four-frequency candidates; validation-selected MLP3/MLP9/Fourier controls,200 iteration cap. Per-task fitting, unequal training budgets; no quantum advantage claim.','reviewed':'40_checkpoint: wave gains did not establish breadth; radial extrapolation fails majority control.','metrics':'accuracy,balanced accuracy,cross entropy and majority baseline; retain optimizer failures.'})
 rows=[];fits=[]
 for seed in range(3):
  X=np.random.default_rng(131000+seed).uniform(-np.pi,np.pi,(1408,2))
  for task in TASKS41:
   y=target41(X,task);tr=(X[:128],y[:128]);va=(X[128:384],y[128:384]);T,t=X[384:],y[384:]
   bs=[fit_readout(tr,va,f) for f in ([1.],[.5,1.,2.,3.])];ls=[joint_fit(tr,va,b,seed) for b in bs];b=min(bs,key=lambda f:f['validation_loss']);l=min(ls,key=lambda f:f['validation_loss'])
   cs=[]
   for name in ('MLP_3','MLP_9','fourier'):
    c,_=fit_model(name,tr,va,seed,maxiter=200);cs.append({**c,'model':name})
   c=min(cs,key=lambda f:f['validation_loss']);fits.append(dict(seed=seed,task=task,baseline=b,learned=l,all_learned_candidates=ls,classical=c,classical_candidates=cs))
   for name,z in [('fixed',learned_predict(T,b)),('learned',learned_predict(T,l)),('classical',classical_predict(T,c))]:rows.append(dict(seed=seed,task=task,model=name,**richer_metrics(z,t)))
   print(seed,task,flush=True)
 log('41_result',{'rows':rows,'fits':fits,'decision':'Compare paired per-seed breadth against classical controls before measurement-noise readout training; preserve all task failures.'})
if __name__=='__main__':main()
