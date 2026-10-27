"""Experiment21: uncertainty, abstention and finite-shot robustness."""
from qnn_mixture import *

def stats(z,y,threshold):
 p=expit(z);confidence=np.maximum(p,1-p);correct=(p>.5)==y;ece=0
 for lo,hi in zip(np.linspace(.5,1,11)[:-1],np.linspace(.5,1,11)[1:]):
  mask=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
  if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
 keep=confidence>=threshold
 return {**evaluate(z,y),'ece10':float(ece),'confident_wrong_fraction':float(np.mean((confidence>=.9)&~correct)),'coverage':float(keep.mean()),'accepted_accuracy':float(correct[keep].mean()) if keep.any() else None}

def main():
 review();log('21_plan',{'hypothesis':'High average accuracy need not mean trustworthy confidence. Test calibration, selective prediction, input corruption and actual finite-measurement uncertainty.','design':'Frozen128-sample models,3 seeds x4 tasks. Temperature fit and confidence threshold selected only on128 validation points.512 test points.','selective_rule':'Choose greatest validation coverage among thresholds.5..99 with>=90% validation accuracy and>=16 accepted examples; otherwise abstain on everything. This is a heuristic, not a certified guarantee.','corruption':'Gaussian input noise sigma0/.3/.8 radians; labels remain those of clean inputs. Noisy observations can cross true boundaries; not all resulting error is model failure.','measurements':'QNN mixture measured on two circuits, total256/1024 shots split equally. Fits were trained on ideal expectations; temperature also from ideal validation.','models':'QNN mixture,9-parameter MLP,33-parameter MLP,exact-feature classical control; same labels/inputs, no transformer or general intelligence claim.'})
 mix=json.loads((LOG/'20_result.json').read_text())['fits'];basic=json.loads((LOG/'16_result.json').read_text())['fits'];matched=json.loads((LOG/'18_result.json').read_text())['fits'];rows=[];cal=[]
 for mf in mix:
  if mf['n_train']!=128:continue
  seed,task=mf['seed'],mf['task'];_,val,test=splits(seed,task,128)
  predictors={};mp=np.array(mf['parameters']);predictors['QNN_mixture']=lambda X,p=mp:mixture_logit(X,p)
  for model,h in [('MLP_2',2),('MLP_8',8)]:
   f=next(f for f in basic if f['seed']==seed and f['task']==task and f['n_train']==128 and f['model']==model);p=np.array(f['parameters'])
   predictors[model]=lambda X,p=p,h=h:np.tanh((X/np.pi)@p[:2*h].reshape(2,h)+p[2*h:3*h])@p[3*h:4*h]+p[-1]
  f=next(f for f in matched if f['seed']==seed and f['task']==task and f['n_train']==128);p=np.array(f['parameters']);predictors['feature_control']=lambda X,p=p:p[0]*np.cos(X[:,0])*np.cos(X[:,1])+p[1]*np.sin(X[:,1])+p[2]
  for model,predict in predictors.items():
   zv=predict(val[0]);yv=val[1]
   objective=lambda a:np.mean(np.logaddexp(0,zv/np.exp(a[0]))-yv*zv/np.exp(a[0]))
   fit=minimize(objective,[0.],method='L-BFGS-B',bounds=[(-3,3)])
   T=float(np.exp(fit.x[0]));pv=expit(zv/T);cv=np.maximum(pv,1-pv);ok=(pv>.5)==yv
   choices=[t for t in np.linspace(.5,.99,50) if np.sum(cv>=t)>=16 and ok[cv>=t].mean()>=.9]
   threshold=float(min(choices)) if choices else float('inf')
   cal.append({'seed':seed,'task':task,'model':model,'temperature':T,'threshold':threshold if np.isfinite(threshold) else None,'success':bool(fit.success)})
   for sigma in (0.,.3,.8):
    X,y=test;noisyX=X+np.random.default_rng(69000+seed+int(sigma*1000)).normal(0,sigma,X.shape)
    z=predict(noisyX)
    for label,zz in [('raw',z),('calibrated',z/T)]:rows.append({'seed':seed,'task':task,'model':model,'sigma':sigma,'measurements':0,'repeat':0,'calibration':label,**stats(zz,y,threshold if label=='calibrated' else .9)})
    if model=='QNN_mixture':
     a,b=branch_features(noisyX,mp);weight=expit(mp[6]);assert np.all(abs(a)<=1+1e-12) and np.all(abs(b)<=1+1e-12)
     for B in (256,1024):
      for rep in range(10):
       r=np.random.default_rng(70000+seed*1000+B+rep+int(sigma*10000));na=B//2;nb=B-na
       za=2*r.binomial(na,np.clip((1+a)/2,0,1))/na-1;zb=2*r.binomial(nb,np.clip((1+b)/2,0,1))/nb-1
       zz=mp[7]*(weight*za+(1-weight)*zb)+mp[8]
       rows.append({'seed':seed,'task':task,'model':model,'sigma':sigma,'measurements':B,'repeat':rep,'calibration':'calibrated',**stats(zz/T,y,threshold)})
 log('21_result',{'rows':rows,'calibrations':cal,'decision':'Review confidence errors and accepted accuracy on held-out/corrupted data. Temperature scaling cannot improve the decision boundary, and an empirical90% validation threshold is not an OOD safety guarantee.'})

if __name__=='__main__':main()
