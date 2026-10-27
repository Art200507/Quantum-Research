"""Meaningful numerical checks of new circuit/readout and learning calculations."""
from capabilities import *
from qnn_mixture import ACTIVE,branch_features,mixture_logit

def validate():
 r=np.random.default_rng(90000);X=r.uniform(-np.pi,np.pi,(16,2));p=r.normal(size=9);theta=np.zeros(8);theta[ACTIVE]=p[1:6]
 a,b=branch_features(X,p);errors=[]
 for i in range(len(X)):
  state=Statevector.from_instruction(qcirc(X[i],theta,2));z=float(np.dot(abs(state.data)**2,[1,1,-1,-1]));errors.append(abs(z-b[i]))
 assert max(errors)<1e-12
 # Check the output is a classical mixture of the two actual Born expectations.
 w=expit(p[6]);expected=p[7]*(w*a+(1-w)*b)+p[8];assert np.allclose(expected,mixture_logit(X,p))
 # Independently finite-difference the analytic gradient used for the MLP baseline.
 h=2;weights=r.normal(size=9);y=r.integers(0,2,16);A=X/np.pi
 def loss(v):
  H=np.tanh(A@v[:4].reshape(2,2)+v[4:6]);z=H@v[6:8]+v[8]
  return np.mean(np.logaddexp(0,z)-y*z)+1e-3*np.mean(v*v)
 H=np.tanh(A@weights[:4].reshape(2,2)+weights[4:6]);z=H@weights[6:8]+weights[8];dz=(expit(z)-y)/len(y);dh=dz[:,None]*weights[6:8]*(1-H*H)
 analytic=np.r_[(A.T@dh).ravel(),dh.sum(0),H.T@dz,dz.sum()]+2e-3*weights/9
 numerical=[]
 for j in range(9):
  delta=np.zeros(9);delta[j]=1e-5;numerical.append((loss(weights+delta)-loss(weights-delta))/2e-5)
 grad_error=float(np.max(abs(analytic-numerical)));assert grad_error<1e-7
 train,val,test=splits(0,'radial',128);assert len(np.unique(np.concatenate([train[0],val[0],test[0]]),axis=0))==768
 return {'mixture_circuit_error':max(errors),'MLP_gradient_error':grad_error,'disjoint_IID_points':768,'passed':True}

if __name__=='__main__':print(validate())
