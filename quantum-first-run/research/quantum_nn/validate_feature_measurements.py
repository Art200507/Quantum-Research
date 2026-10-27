"""Check physical grouped estimators and independent classical feature control."""
from feature_control import *
from prediction_sets import cutoff,set_metrics

def validate():
 r=np.random.default_rng(119000);X=r.uniform(-np.pi,np.pi,(8,2));frequencies=[.5,1.,2.,3.];F=features(X,frequencies);errors=[]
 for bases in ([a+b for a in 'XYZ' for b in 'XYZ'],['XX','ZZ','XZ','YY']):
  P=basis_probabilities(X,frequencies,bases);exact,_,mask=measure(P,frequencies,bases,None,None)
  error=float(np.max(abs(exact[:,mask]-F[:,mask])));assert error<1e-12;errors.append(error)
  assert np.all(mask[F.std(0)>1e-8])
  for B in (256,1024):
   sampled,used,_=measure(P,frequencies,bases,B,np.random.default_rng(1));assert used==B and np.all(abs(sampled)<=1+1e-12)
 # Sampled feature means converge to physical expectations, with actual group covariance retained.
 P=basis_probabilities(X,[1.],['XX','ZZ','XZ','YY']);truth=features(X,[1.]);draws=np.array([measure(P,[1.],['XX','ZZ','XZ','YY'],256,np.random.default_rng(120000+i))[0] for i in range(500)])
 mean=draws.mean(0);covered=measure(P,[1.],['XX','ZZ','XZ','YY'],None,None)[2]
 exposures=np.array([64*sum(all(a=='I' or a==b for a,b in zip(pauli,basis)) for basis in ['XX','ZZ','XZ','YY']) for pauli in PAULIS])
 se=np.sqrt(np.maximum(0,1-truth**2)/np.maximum(exposures,1)/500)
 assert np.all(abs(mean[:,covered]-truth[:,covered])<=6*se[:,covered]+1e-12)
 # Prediction-set bookkeeping: all/empty sets cannot masquerade as useful single predictions.
 allstats=set_metrics(np.zeros(10),np.arange(10)%2,1.);emptystats=set_metrics(np.zeros(10),np.arange(10)%2,.1)
 assert allstats['marginal_coverage']==1 and allstats['singleton_rate']==0 and allstats['ambiguous_rate']==1
 assert emptystats['marginal_coverage']==0 and emptystats['empty_rate']==1
 z=np.linspace(-4,4,100);y=np.arange(100)%2;assert cutoff(z,y,.05)>=cutoff(z,y,.1)
 return {'passed':True,'exact_group_reconstruction_errors':errors,'elementary_classical_feature_error':float(np.max(abs(F-elementary_features(X,frequencies)))),'Monte_Carlo_replicates':500,'measurement_mean_check':'all covered features within6 exact-marginal Monte Carlo SE','validation_refinement':'Initially using empirical SE falsely failed a nearly deterministic observable:500 replicates saw no rare flips, yielding zero sample variance. Use the known Born marginal variance for this simulation check; estimator and experiment results unchanged.','budgets':[256,1024],'set_bookkeeping':'all/empty/monotone cases pass'}

if __name__=='__main__':print(validate())
