"""Experiment36: frozen learning protocol, new directions and classical controls."""
from learned_inputs import *
DIRECTIONS=((1.3,-.7),(.6,1.9),(-1.1,1.5),(2.2,.4))

def main():
 review();log('36_plan',{'hypothesis':'Learned input directions should help more than the particular diagonal wave that motivated the change.','design':'Four prelisted direction vectors,3 fresh seeds,128 train/256 validation/1024 test. Same frequency choices,regularization grid,two encoder starts,200 iteration cap as35. No task-specific initial direction supplied.','directions':DIRECTIONS,'controls':'Fixed quantum-feature encoder and validation-selected classical MLP3/MLP9/Fourier. MLP3/9 have13/37 parameters matching learned single/multi head parameter counts, though function classes and training costs differ.','qualification':'Exploratory confirmation within a wave task family, not arbitrary tasks or broad quantum intelligence. Optimizer failures retained.'})
 rows=[];fits=[]
 for seed in range(3):
  X=np.random.default_rng(123000+seed).uniform(-np.pi,np.pi,(1408,2))
  for i,direction in enumerate(DIRECTIONS):
   y=(np.sin(X@np.array(direction))>0).astype(float);train=(X[:128],y[:128]);val=(X[128:384],y[128:384]);Xt,yt=X[384:],y[384:]
   fixed=[fit_readout(train,val,f) for f in ([1.],[.5,1.,2.,3.])];learned=[joint_fit(train,val,b,seed) for b in fixed]
   b=min(fixed,key=lambda f:f['validation_loss']);l=min(learned,key=lambda f:f['validation_loss'])
   controls=[]
   for model in ('MLP_3','MLP_9','fourier'):
    fit,predict=fit_model(model,train,val,seed,maxiter=200);controls.append({**fit,'model':model})
   c=min(controls,key=lambda f:f['validation_loss']);fits.append({'seed':seed,'task':i,'direction':direction,'baseline':b,'learned':l,'all_learned_candidates':learned,'classical':c,'classical_candidates':controls})
   for model,z in [('fixed_inputs',learned_predict(Xt,b)),('learned_inputs',learned_predict(Xt,l)),('selected_classical',classical_predict(Xt,c))]:rows.append({'seed':seed,'direction':i,'model':model,**richer_metrics(z,yt)})
 log('36_result',{'rows':rows,'fits':fits,'decision':'Inspect fresh directions and optimization failures; confirm improvement is not restricted to one example. Next test measured inference and selected checkpoints, not only ideal outcomes.'})

if __name__=='__main__':main()
