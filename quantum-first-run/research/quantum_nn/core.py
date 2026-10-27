"""Actual two-qubit circuit models with a checked vectorized local simulator."""
from pathlib import Path
import json,os,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,softmax
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
ROOT=Path(__file__).parent;LOG=Path(os.environ.get('QNN_LOG_DIR',str(ROOT/'logs')));LOG.mkdir(parents=True,exist_ok=True)

def log(name,payload):
 path=LOG/(name+'.json')
 if path.exists():raise FileExistsError(path)
 path.write_text(json.dumps(payload,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
 print(json.dumps(payload,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())[:9000])

def review():
 names=sorted(p for p in LOG.glob('*.json') if p.name.endswith(('_result.json','_checkpoint.json')))
 if names:
  p=json.loads(names[-1].read_text());print('Reviewed previous log:',names[-1].name,p.get('decision',p.get('hypothesis','')))

def ry(t):
 return np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]],complex)
def apply(S,U,q):return S@ (np.kron(np.eye(2),U) if q==0 else np.kron(U,np.eye(2))).T

def encoded(X):
 a=X[:,0]/2;b=X[:,1]/2
 return np.stack([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)*np.sin(b)],1).astype(complex)

def circuit(x,p,attention=False,broken=False,reupload_scaled=False):
 qc=QuantumCircuit(2)
 a=p[:2] if attention else [1.,1.]
 qc.ry(a[0]*x[0],0);qc.ry(a[1]*x[1],1);qc.cx(0,1)
 theta=p[2:4] if attention else p[:2]
 qc.ry(theta[0],0);qc.ry(theta[1],1)
 if attention and not broken:
  qc.rz(x[0]*(a[0] if reupload_scaled else 1),0);qc.rz(x[1]*(a[1] if reupload_scaled else 1),1);qc.cx(1,0)
 return qc

def states(X,p,attention=False,broken=False,reupload_scaled=False):
 a=p[:2] if attention else np.ones(2);S=encoded(X*a)
 S=S[:,[0,3,2,1]] # CX control 0 -> target 1
 theta=p[2:4] if attention else p[:2]
 S=apply(apply(S,ry(theta[0]),0),ry(theta[1]),1)
 if attention and not broken:
  z=np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
  S*=np.exp(-.5j*((X*a if reupload_scaled else X)@z.T));S=S[:,[0,1,3,2]] # CX control 1 -> target 0
 return S

def check():
 r=np.random.default_rng(19);X=r.normal(size=(8,2));p=np.array([.8,.3,.4,-.6]);err=0
 for att,broken in [(False,False),(True,False),(True,True)]:
  S=states(X,p,att,broken)
  for x,s in zip(X,S):err=max(err,float(np.max(abs(s-Statevector.from_instruction(circuit(x,p,att,broken)).data))))
 assert err<1e-12
 return err

def qnn_z(X,theta):
 S=states(X,theta);return abs(S)**2@np.array([1,1,-1,-1])

def fit_logistic(X,y):
 def loss(p):
  z=X@p[:-1]+p[-1];return np.mean(np.logaddexp(0,z)-y*z)+1e-4*np.sum(p[:-1]**2)
 return minimize(loss,np.zeros(X.shape[1]+1),method='L-BFGS-B',options={'maxiter':150})

def episodes(seed,n,m=8):
 r=np.random.default_rng(seed);Q=r.uniform(-1,1,(n,2));K=r.uniform(-1,1,(n,m,2));delta=K-Q[:,None,:]
 target=(delta[:,:,0]**2+.05*delta[:,:,1]**2).argmin(1)
 return Q,K,target

def overlaps(Q,K,p,broken=False,reupload_scaled=False):
 qs=states(Q,p,True,broken,reupload_scaled);ks=states(K.reshape(-1,2),p,True,broken,reupload_scaled).reshape(*K.shape[:2],4)
 return np.clip(abs(np.einsum('ni,nmi->nm',qs.conj(),ks))**2,0,1)

if __name__=='__main__':print(check())
