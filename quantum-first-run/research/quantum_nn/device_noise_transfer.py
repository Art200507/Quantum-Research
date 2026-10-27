"""47: frozen readout transfer under explicit toy device channels."""
from confidence_control import *
from measured_features import ROT
from qiskit.quantum_info import DensityMatrix,Operator
from qiskit_aer.noise import depolarizing_error
ENVIRONMENTS=((0.,0.),(.02,.01),(.10,.03))

def transition(r):
 return np.array([[(r**((i^j).bit_count()))*((1-r)**(2-(i^j).bit_count())) for j in range(4)] for i in range(4)])
def channel_probabilities(X,fit,p,r):
 P=basis_probabilities(X@np.array(fit['input_matrix']),fit['frequencies'],BASES);M=transition(r)
 return [((1-p)*a+p/4)@M for a in P]
def validate():
 X=np.random.default_rng(147000).normal(size=(3,2));fit={'input_matrix':np.array([[1.,.2],[-.3,.8]]),'frequencies':[.5,2.]};error=0.
 for p,r in ENVIRONMENTS:
  actual=channel_probabilities(X,fit,p,r)
  for v,f in enumerate(fit['frequencies']):
   for i,x in enumerate(X@fit['input_matrix']):
    rho=DensityMatrix.from_instruction(map_circuit(x,f)).evolve(depolarizing_error(p,2).to_quantumchannel())
    for k,b in enumerate(BASES):
     q=rho.evolve(Operator(np.kron(ROT[b[0]],ROT[b[1]]))).probabilities();q=q@transition(r);error=max(error,float(abs(q-actual[v*4+k][i]).max()))
 assert error<1e-12;return error

def main():
 review();log('47_plan',{'reviewed':'46_result: temperature lowers overconfidence without altering accuracy; noise refit improves low-shot decisions,but high-shot temperature has better CE.','hypothesis':'Shot-noise-trained readouts may fail when systematic device errors shift observable means.','design':'Freeze45 learned encoders,noise readouts and46 temperature scalars. Fresh1024test points per5 training seeds,5 paired measurement repeats at256/1024shots,three prelisted channels. No device-error-aware retraining/calibration.','channels':ENVIRONMENTS,'definition':'Global2qubit depolarization rho->(1-p)rho+pI/4 applied once afterCX;independent symmetric readout bit flip probability r perqubit. Ideal basis rotations. Exact Born probabilities transformed before grouped multinomial draws.','validation':'Independent Qiskit Aer density-matrix channel and basis rotations compared against fast probability transformation.','scope':'Hypothetical stress test,not IBM-calibrated device model;no depth dependence,crosstalk,drift or per-gate errors;no hardware extrapolation claim.'})
 error=validate();old=json.loads((LOG/'45_result.json').read_text());temps=json.loads((LOG/'46_result.json').read_text());rows=[]
 for item in old['fits']:
  s,B,base=item['seed'],item['budget'],item['base'];a=next(f['inverse_temperature'] for f in temps['fits'] if f['seed']==s and f['budget']==B);T=np.random.default_rng(148000+s).uniform(-np.pi,np.pi,(1024,2));y=target41(T,'annulus')
  for env,(p,r) in enumerate(ENVIRONMENTS):
   P=channel_probabilities(T,base,p,r)
   for rep in range(5):
    F,used,cov=measure(P,base['frequencies'],BASES,B,np.random.default_rng(149000+s*10000+B+100*env+rep));assert used==B and np.all(cov[np.array(base['mask'],bool)]);z=learned_predict(T,base,F);zn=learned_predict(T,item['noise_fit'],F)
    for name,zz in [('ideal',z),('temperature',a*z),('noise',zn)]:rows.append(dict(seed=s,budget=B,depolarization=p,readout_flip=r,repeat=rep,model=name,**metrics(zz,y)))
 log('47_result',{'rows':rows,'channel_Qiskit_max_error':error,'decision':'Review stress-test transfer and keep channel mismatch visible. Next compare matched device-aware calibration/refitting and equal-shot grouping controls;hardware proposal requires realistic backend channels and broader tasks.'})
if __name__=='__main__':main()
