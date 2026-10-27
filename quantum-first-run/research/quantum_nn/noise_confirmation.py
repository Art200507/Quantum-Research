"""43: fresh-cohort confirmation of the observed ring reliability lead."""
from noise_aware_readout import *
from measured_features import metrics

def main():
 review();log('43_plan',{'hypothesis':'Stage42 ring-task probability improvement at256 shots may repeat on new samples.','reviewed':'42_result: annulus CE .5475 to .3386 at256shots; accuracy .8430 to .8517. At1024shots noise training slightly hurts. Other tasks mixed.','design':'Exploratory follow-up selected after42; freeze all3 ring encoders and noise-trained readouts. Fresh2048 test points per seed,10 measurement repeats at256 and1024shots. Paired identical measurements. No retuning. Report per-seed means and overconfidence errors; repeats are not independent training seeds.','scope':'Same synthetic distribution; not independent task-family replication,hardware noise or quantum advantage.'})
 old=json.loads((LOG/'41_result.json').read_text());trained=json.loads((LOG/'42_result.json').read_text());rows=[]
 for seed in range(3):
  base=next(f['learned'] for f in old['fits'] if f['seed']==seed and f['task']=='annulus');T=np.random.default_rng(134000+seed).uniform(-np.pi,np.pi,(2048,2));y=target41(T,'annulus')
  for B in (256,1024):
   fit=next(f['fit'] for f in trained['fits'] if f['seed']==seed and f['task']=='annulus' and f['budget']==B)
   for rep in range(10):
    F=noisy_features(T,base,B,np.random.default_rng(135000+seed*10000+B+rep))
    for name,f in [('ideal_training',base),('noise_training',fit)]:rows.append(dict(seed=seed,budget=B,repeat=rep,model=name,**metrics(learned_predict(T,f,F),y)))
 log('43_result',{'rows':rows,'decision':'Judge probability reliability and accuracy separately, retain high-shot regression, and seek larger independent task/seed replication before recommending noise-aware training generally.'})
if __name__=='__main__':main()
