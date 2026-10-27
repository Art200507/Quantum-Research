"""45: independent training cohorts for ring measurement reliability."""
from noise_aware_readout import *
from measured_features import metrics

def main():
 review();log('45_plan',{'reviewed':'44_checkpoint and43_result: ring low-shot confidence improvement used only3 original training seeds; high-shot regression retained.','hypothesis':'Low-shot ring reliability improvement survives independent retraining.','design':'5 new training/encoder-start seeds 10..14;128train/256validation; original frequency choices,two starts,200 iterations; choose learned encoder by validation only. Refit noisy readout with4 measurement draws per point at256/1024shots; fresh2048test,10 paired measurement repeats.','primary':'Paired cross entropy per training seed at256shots; also Brier,accuracy and confidently-wrong fraction,plus1024shot condition. No seed exclusions.','cost':'Per seed/budget:512*B train,1024*B validation,20480*B test shots. Encoder training ideal. Repeated measurements are not independent training seeds.','scope':'Same synthetic ring distribution,2qubits,classically reproducible,no gate noise.'})
 rows=[];fits=[]
 for seed in range(10,15):
  X=np.random.default_rng(141000+seed).uniform(-np.pi,np.pi,(384,2));y=target41(X,'annulus');tr=(X[:128],y[:128]);va=(X[128:],y[128:]);bs=[fit_readout(tr,va,f) for f in ([1.],[.5,1.,2.,3.])];ls=[joint_fit(tr,va,b,seed) for b in bs];base=min(ls,key=lambda f:f['validation_loss'])
  T=np.random.default_rng(142000+seed).uniform(-np.pi,np.pi,(2048,2));ty=target41(T,'annulus')
  for B in (256,1024):
   r=np.random.default_rng(143000+seed*10000+B);F=np.concatenate([noisy_features(tr[0],base,B,r) for _ in range(4)]);V=np.concatenate([noisy_features(va[0],base,B,r) for _ in range(4)]);fit=refit(F,np.tile(tr[1],4),V,np.tile(va[1],4),base)
   fits.append(dict(seed=seed,budget=B,base=base,noise_fit=fit,encoder_candidates=ls,validation_features=V,validation_labels=np.tile(va[1],4)))
   for rep in range(10):
    Ft=noisy_features(T,base,B,np.random.default_rng(144000+seed*10000+B+rep))
    for name,f in [('ideal',base),('noise',fit)]:rows.append(dict(seed=seed,budget=B,repeat=rep,model=name,**metrics(learned_predict(T,f,Ft),ty)))
  print('Completed seed',seed,flush=True)
 log('45_result',{'rows':rows,'fits':fits,'decision':'Review independent seed-level improvements,then compare against a simpler confidence-only calibration control before attributing gains to richer noise training.'})
if __name__=='__main__':main()
