"""Experiment30: separate calibration data and useful uncertainty, not confidence alone."""
from select_features import *

def cutoff(z,y,alpha=.1):
 p=expit(z);scores=np.where(y==1,1-p,p);k=int(np.ceil((len(y)+1)*(1-alpha)))
 return float(np.sort(scores)[k-1]) if k<=len(y) else float('inf')

def set_metrics(z,y,q):
 p=expit(z);sets=np.c_[p<=q,1-p<=q];sizes=sets.sum(1);covered=sets[np.arange(len(y)),y.astype(int)];single=sizes==1
 return {'marginal_coverage':float(covered.mean()),'singleton_rate':float(single.mean()),'singleton_accuracy':float(covered[single].mean()) if single.any() else None,'ambiguous_rate':float(np.mean(sizes==2)),'empty_rate':float(np.mean(sizes==0)),'accuracy':float(np.mean((z>0)==y))}

def sampled_z(X,fit,seed):
 bases=['XX','ZZ','XZ','YY'];probs=basis_probabilities(X,fit['frequencies'],bases);F,used,_=measure(probs,fit['frequencies'],bases,1024,np.random.default_rng(seed));assert used==1024
 return readout(X,fit,F)

def main():
 review();log('30_plan',{'hypothesis':'A separate prediction-set calibration may control marginal errors on matching data while failing under corruption. Must report ambiguous/empty answers, not call coverage intelligence.','design':'Freeze model choices from29.256 entirely new calibration points and512 new test points per seed;5 seeds,4 tasks, quantum and selected classical controls.','quantum_protocols':'Ideal calibration/ideal test; ideal calibration/1024-shot test;1024-shot calibration/1024-shot test. No gate noise. Test input sigma0/.8; labels always from clean inputs.','separation':'Calibration is not the earlier model-selection validation. Quantile rank ceil((n+1)*.9), score1-p_true. Classical conformal wrapper is not a uniquely quantum algorithm.','cost':'Quantum matched calibration requires256*1024 shots per fitted task/seed; not free.5 measurement repeats are conditional on one calibration cohort per seed.','claims':'Only exchangeable matching protocols support standard marginal-coverage interpretation. No conditional correctness or distribution-shift guarantee; count singleton/ambiguous/empty rates explicitly.'})
 selected=json.loads((LOG/'29_result.json').read_text())['selected'];rows=[];calibrations=[]
 for item in selected:
  seed,task=item['seed'],item['task'];qfit=item['quantum_fit'];cfit=item['classical_fit']
  Xc=np.random.default_rng(91000+seed).uniform(-np.pi,np.pi,(256,2));yc=labels(Xc,task)
  Xt=np.random.default_rng(92000+seed).uniform(-np.pi,np.pi,(512,2));yt=labels(Xt,task)
  qideal=cutoff(readout(Xc,qfit),yc);qsample=cutoff(sampled_z(Xc,qfit,93000+seed),yc);qc=cutoff(classical_predict(Xc,cfit),yc)
  calibrations.append({'seed':seed,'task':task,'quantum_ideal_cutoff':qideal,'quantum_measured_cutoff':qsample,'classical_cutoff':qc})
  for sigma in (0.,.8):
   X=Xt+np.random.default_rng(94000+seed).normal(0,sigma,Xt.shape)
   for model,z,q in [('quantum_ideal',readout(X,qfit),qideal),('classical',classical_predict(X,cfit),qc)]:rows.append({'seed':seed,'task':task,'model':model,'sigma':sigma,'repeat':0,**set_metrics(z,yt,q)})
   for rep in range(5):
    z=sampled_z(X,qfit,95000+seed*100+rep)
    for name,q in [('quantum_measured_clean_calibration',qideal),('quantum_measured_matched_calibration',qsample)]:rows.append({'seed':seed,'task':task,'model':name,'sigma':sigma,'repeat':rep,**set_metrics(z,yt,q)})
 # Set inclusion must be monotone when demanded coverage increases.
 z=np.linspace(-4,4,101);y=np.random.default_rng(1).integers(0,2,101);assert cutoff(z,y,.05)>=cutoff(z,y,.1)
 log('30_result',{'rows':rows,'calibrations':calibrations,'decision':'Review empirical coverage together with useful single-answer rate. If corruption breaks coverage, test matched-corruption calibration explicitly and charge the ambiguity/calibration costs.'})

if __name__=='__main__':main()
