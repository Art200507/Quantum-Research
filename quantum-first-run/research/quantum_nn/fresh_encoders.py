"""62: fresh independent encoder cohorts before a temporal budget benchmark."""
from calibration_budget import *
FRESH_SEEDS=(200,201,202)

def main():
 review()
 log('62_plan',{
  'reviewed':'61_checkpoint and60_result: calibration budgets affect probability error; temporal continuation still reused old encoders.',
  'hypothesis':'Retraining on independent data/start seeds provides a stronger basis for testing calibration budgets under drift.',
  'design':'Three new training seeds, all four non-wave tasks. 128 train,256 encoder-selection validation,1024 held-out ideal test points. Frozen stage35 single/four-frequency protocol,two starts,200 iterations. Select fixed/learned models by validation CE only. No test-dependent exclusions.',
  'scope':'These are genuinely new encoder training cohorts,not merely new measurement draws. Still synthetic two-qubit models with exact classical equivalents. Calibration and drift test data will be generated independently in63.'})
 fits=[];rows=[];errors=[]
 for seed in FRESH_SEEDS:
  X=np.random.default_rng(np.random.SeedSequence([62,seed])).uniform(-np.pi,np.pi,(1408,2))
  for task in TASKS41:
   y=target41(X,task);tr=(X[:128],y[:128]);va=(X[128:384],y[128:384])
   fixed=[fit_readout(tr,va,f) for f in ([1.],[.5,1.,2.,3.])]
   learned=[joint_fit(tr,va,b,seed) for b in fixed]
   b=min(fixed,key=lambda f:f['validation_loss']);l=min(learned,key=lambda f:f['validation_loss'])
   fits.append(dict(seed=seed,task=task,base=l,baseline=b,candidates=learned))
   for name,fit in [('fixed',b),('learned',l)]:rows.append(dict(seed=seed,task=task,model=name,**metrics(learned_predict(X[384:],fit),y[384:])))
   point=X[384]@np.array(l['input_matrix']);F=elementary_features(point[None,:],l['frequencies'])
   for j,f in enumerate(l['frequencies']):
    S=Statevector.from_instruction(map_circuit(point,f));actual=np.array([S.expectation_value(Pauli(p)).real for p in PAULIS]);errors.append(float(abs(F[0,15*j:15*(j+1)]-actual).max()))
   print(seed,task,flush=True)
 assert max(errors)<1e-12
 log('62_result',{'rows':rows,'fits':fits,'circuit_errors':errors,'decision':'Review every task and optimizer outcome;retain all12 selected encoders for the temporal budget test regardless of ideal score.'})
if __name__=='__main__':main()
