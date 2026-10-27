"""Experiment32: independent elementary classical representation of the feature bank."""
from select_features import *

def elementary_features(X,frequencies):
 blocks=[]
 for f in frequencies:
  a,b=(X*f).T;sa,ca,sb,cb=np.sin(a),np.cos(a),np.sin(b),np.cos(b)
  columns={'IX':sa*sb,'IZ':ca,'XI':sb,'XX':sa,'XZ':ca*sb,'YY':-sa*cb,'ZI':ca*cb,'ZZ':cb}
  blocks.append(np.column_stack([columns.get(p,np.zeros(len(X))) for p in PAULIS]))
 return np.concatenate(blocks,1)

def main():
 review();log('32_plan',{'hypothesis':'Improved quantum-feature capability may be expressible by cheap classical trigonometric features in this small circuit. Verify independently before attributing gains to quantum computing.','method':'Conjugate each Pauli by the single CX and evaluate input product-state Bloch coordinates sin/cos. No statevectors in elementary control. Test every trained feature model and selected readout on independent inputs.','scope':'Exact control for this specific2-qubit fixed encoder only; not a claim that arbitrary QNNs are efficiently classically simulable.'})
 X=np.random.default_rng(103000).uniform(-5,5,(1024,2));errors=[];logit_errors=[]
 for stage in (25,26):
  fits=[f for f in json.loads((LOG/f'{stage}_result.json').read_text())['fits'] if f['model']=='quantum_features']
  for fit in fits:
   F=elementary_features(X,fit['frequencies']);errors.append(float(np.max(abs(F-features(X,fit['frequencies'])))))
   logit_errors.append(float(np.max(abs(readout(X,fit,F)-readout(X,fit)))))
 assert max(errors)<1e-12 and max(logit_errors)<1e-9
 log('32_result',{'feature_error':max(errors),'logit_error':max(logit_errors),'models_checked':len(logit_errors),'decision':'Retain improved observable/readout architecture as a QNN benchmark, but its gains are not uniquely quantum. Test fixed architecture on new task families without changing its frequencies.'})

if __name__=='__main__':main()
