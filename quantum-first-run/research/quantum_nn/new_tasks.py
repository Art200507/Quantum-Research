"""Experiment33: freeze encoder design and challenge with new task families."""
from select_features import *
NEWTASKS=('rotated_ellipse','diagonal_wave','annulus','tilted_interaction')

def new_labels(X,task):
 a,b=X.T
 if task=='rotated_ellipse':f=((.8*a+.6*b)/2.7)**2+((-.6*a+.8*b)/1.5)**2-1
 elif task=='diagonal_wave':f=np.sin(1.7*a+.9*b)
 elif task=='annulus':f=np.minimum((a*a+b*b)-1.2**2,2.6**2-(a*a+b*b))
 elif task=='tilted_interaction':f=np.sin(a)*np.cos(1.4*b)+.35*a
 else:raise ValueError(task)
 return (f>0).astype(float)

def richer_metrics(z,y):
 pred=z>0;parts=[np.mean(pred[y==c]==c) for c in (0,1) if np.any(y==c)]
 return {**evaluate(z,y),'balanced_accuracy':float(np.mean(parts)),'positive_label_fraction':float(y.mean()),'majority_accuracy':float(max(y.mean(),1-y.mean()))}

def main():
 review();log('33_plan',{'hypothesis':'The feature-bank design may have been adapted to the original benchmark. Test new task families with encoder frequencies frozen and no new basis engineering.','tasks':NEWTASKS,'design':'5 new seeds x4 tasks,128 train/256 validation/1024 test. Choose single/multi-scale readouts and classical MLP8/Fourier using validation only. Frequencies stay1 or.5/1/2/3; original regularization grid unchanged.','measurements':'Selected quantum readout also evaluated with1024 total shots over4 bases/view,5 repeats. No gate noise.','metrics':'Accuracy,balanced accuracy,majority baseline,probability loss. These are further synthetic tasks, not real-world capability validation.'})
 rows=[];fits=[]
 for seed in range(5):
  X=np.random.default_rng(110000+seed).uniform(-np.pi,np.pi,(1408,2))
  for task in NEWTASKS:
   y=new_labels(X,task);train=(X[:128],y[:128]);val=(X[128:384],y[128:384]);test=(X[384:],y[384:])
   qfits=[fit_readout(train,val,f) for f in ([1.],[.5,1.,2.,3.])];qfit=min(qfits,key=lambda f:f['validation_loss'])
   cfits=[]
   for model in ('MLP_8','fourier'):
    fit,predict=fit_model(model,train,val,seed);cfits.append({**fit,'model':model})
   cfit=min(cfits,key=lambda f:f['validation_loss']);fits.append({'seed':seed,'task':task,'quantum_fit':qfit,'classical_fit':cfit,'quantum_candidates':qfits,'classical_candidates':cfits})
   Xt,yt=test
   for model,z in [('selected_quantum',readout(Xt,qfit)),('selected_classical',classical_predict(Xt,cfit))]:rows.append({'seed':seed,'task':task,'model':model,'budget':0,'repeat':0,**richer_metrics(z,yt)})
   probs=basis_probabilities(Xt,qfit['frequencies'],['XX','ZZ','XZ','YY'])
   for rep in range(5):
    F,used,_=measure(probs,qfit['frequencies'],['XX','ZZ','XZ','YY'],1024,np.random.default_rng(111000+seed*100+rep));assert used==1024
    rows.append({'seed':seed,'task':task,'model':'selected_quantum','budget':used,'repeat':rep,**richer_metrics(readout(Xt,qfit,F),yt)})
 log('33_result',{'rows':rows,'fits':fits,'decision':'Inspect new-task failures before modifying encoding. Any future feature-learning change must be tested on another untouched task cohort; keep all current negative findings.'})

if __name__=='__main__':main()
