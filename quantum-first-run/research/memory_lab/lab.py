"""Bounded classical associative-memory experiments. No remote execution.

Phase binding: m = sum_i k_i * v[label_i]; retrieve Re((conj(q)*m) @ conj(V).T)/d.
All phase arithmetic here is classical. V is a shared decoder codebook, charged separately.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, time
import numpy as np
from scipy.linalg import solve
ROOT=Path(__file__).parent
RESULTS=Path(os.environ.get('MEMORY_LAB_RESULTS',str(ROOT/'results')))
RESULTS.mkdir(parents=True,exist_ok=True)

def data(seed,n,d,c=16,kind='phase'):
 r=np.random.default_rng(seed)
 if kind=='phase':
  K=np.exp(1j*r.uniform(-np.pi,np.pi,(n,d))).astype(np.complex64)
  V=np.exp(1j*r.uniform(-np.pi,np.pi,(c,d))).astype(np.complex64)
 else:
  K=r.choice(np.array([-1,1],dtype=np.float32),(n,d))
  V=r.choice(np.array([-1,1],dtype=np.float32),(c,d))
 y=r.integers(c,size=n)
 return K,V,y,r

def noisy(K,r,sigma):
 return K*np.exp(1j*r.normal(0,sigma,K.shape)).astype(np.complex64) if np.iscomplexobj(K) else K*r.choice(np.array([-1,1],dtype=np.float32),K.shape,p=[sigma,1-sigma])

def int8_state(A):
 if np.iscomplexobj(A):
  scale=np.float32(max(abs(A.real).max(),abs(A.imag).max())/127 or 1)
  re=np.round(A.real/scale).astype(np.int8);im=np.round(A.imag/scale).astype(np.int8)
  return (re.astype(np.float32)+1j*im.astype(np.float32))*scale,re.nbytes+im.nbytes+4
 scale=np.float32(abs(A).max()/127 or 1);payload=np.round(A/scale).astype(np.int8)
 return payload.astype(np.float32)*scale,payload.nbytes+4

def scores(K,V,y,Q,method='bind',banks=1,quant=False,ridge=.1):
 n,d=K.shape;c=len(V); state=None
 if method=='bind':
  state=np.sum(K*V[y],axis=0)
  if quant:
   # Store integer payload + scales; decoder reconstruction is temporary query workspace.
   if np.iscomplexobj(state):
    scale=max(abs(state.real).max(),abs(state.imag).max())/127 or 1
    real=np.round(state.real/scale).astype(np.int8);imag=np.round(state.imag/scale).astype(np.int8)
    state=(real.astype(np.float32)+1j*imag.astype(np.float32))*scale
    stored=real.nbytes+imag.nbytes+4
   else:
    scale=abs(state).max()/127 or 1;payload=np.round(state/scale).astype(np.int8)
    state=payload.astype(np.float32)*scale;stored=payload.nbytes+4
  else:stored=state.nbytes
  S=((Q.conj()*state)@V.conj().T).real/d
  decoder=V.nbytes
 elif method=='exact':
  state=K.copy();Skey=(Q@state.conj().T).real/d
  # Hard nearest-key retrieval, a stronger specialized baseline than soft attention here.
  S=np.eye(c,dtype=np.float32)[y[Skey.argmax(axis=1)]]
  stored=state.nbytes+y.astype(np.int32).nbytes;decoder=0
 elif method=='linear':
  Y=np.eye(c,dtype=np.float32)[y]
  state=K.conj().T@Y/d
  stored=state.nbytes
  if quant:state,stored=int8_state(state)
  S=(Q@state).real;decoder=0
 elif method in ('ridge','ridge32'):
  X=np.concatenate([K.real,K.imag],axis=1) if np.iscomplexobj(K) else K
  Xq=np.concatenate([Q.real,Q.imag],axis=1) if np.iscomplexobj(Q) else Q
  X=X/np.sqrt(d);Xq=Xq/np.sqrt(d)
  if method=='ridge32':X=X.astype(np.float32);Xq=Xq.astype(np.float32)
  Y=np.eye(c,dtype=np.float32)[y]
  state=solve(X.T@X+ridge*np.eye(X.shape[1],dtype=np.float32),X.T@Y,assume_a='pos')
  stored=state.nbytes
  if quant:state,stored=int8_state(state)
  S=Xq@state;decoder=0
 elif method=='bank':
  # Deterministic content routing from signs of first log2(banks) real coordinates.
  bits=int(np.log2(banks));weight=2**np.arange(bits)
  route=lambda x: ((x[:,:bits].real>0)*weight).sum(axis=1)
  ids=route(K);qid=route(Q)
  state=np.zeros((banks,d),dtype=K.dtype)
  np.add.at(state,ids,K*V[y])
  S=((Q.conj()*state[qid])@V.conj().T).real/d
  stored=state.nbytes;decoder=V.nbytes
 else:raise ValueError(method)
 return S,{'state_bytes':int(stored),'decoder_bytes':int(decoder),'total_persistent_bytes':int(stored+decoder)}

def trial(seed,n,d,kind='phase',sigma=0.,method='bind',**kwargs):
 K,V,y,r=data(seed,n,d,kind=kind);Q=noisy(K,r,sigma) if sigma else K.copy()
 start=time.perf_counter();S,cost=scores(K,V,y,Q,method,**kwargs);elapsed=time.perf_counter()-start
 return dict(seed=seed,n=n,d=d,kind=kind,sigma=sigma,method=method,accuracy=float(np.mean(S.argmax(axis=1)==y)),seconds_build_and_batch_query=elapsed,**cost,**kwargs)

def summarize(rows):
 groups={}
 for x in rows:
  key=tuple((k,v) for k,v in x.items() if k not in ('seed','accuracy','seconds_build_and_batch_query'))
  groups.setdefault(key,[]).append(x)
 out=[]
 for key,rs in groups.items():
  a=np.array([x['accuracy'] for x in rs]);out.append({**dict(key),'accuracy_mean':float(a.mean()),'seed_sd':float(a.std(ddof=1)) if len(a)>1 else 0,'seed_se':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0,'seeds':len(a)})
 return out

def save(number,rows,details=None):
 payload={'experiment':number,'rows':rows,'summary':summarize(rows) if rows else [],'details':details or {}}
 path=RESULTS/f'{number:02d}.json'
 if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
 path.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({'experiment':number,'summary':payload['summary'],'details':payload['details']},indent=2))

def checks():
 K,V,y,r=data(1,1,16);S,_=scores(K,V,y,K)
 assert S.argmax()==y[0] and np.isclose(S[0,y[0]],1)
 # Sum-memory invariance to insertion order.
 K,V,y,r=data(1,8,32);a,_=scores(K,V,y,K);b,_=scores(K[::-1],V,y[::-1],K)
 assert np.allclose(a,b,atol=1e-6)
 # Classical real features reproduce phase inner products exactly.
 Z=np.concatenate([K.real,K.imag],axis=1)
 assert np.allclose((K@K.conj().T).real,Z@Z.T,atol=1e-5)
 return ['single-pair exact retrieval','insertion-order invariance','complex-to-real inner-product identity']

def run(e):
 seeds=range(100,112);rows=[]
 if e==1:
  details={'checks':checks(),'purpose':'Verify mechanism at light load; exact nearest-key is a sanity baseline.'}
  for s in seeds:
   for m in ('bind','exact'):rows.append(trial(s,16,128,method=m))
 elif e==2:
  details={'purpose':'Increase load at fixed 128 complex components after light-load check.'}
  for s in seeds:
   for n in (16,64,128,256):
    for m in ('bind','exact'):rows.append(trial(s,n,128,method=m))
 elif e==3:
  details={'purpose':'Equal binding-state bytes: phase d=64 complex64 versus real d=128 float32; decoder bytes also match. Alternate random encodings, same labels per seed are not guaranteed; compare seed means.'}
  for s in seeds:
   for n in (32,128,256):
    for kind,d in [('phase',64),('bipolar',128)]:rows.append(trial(s,n,d,kind=kind))
 elif e==4:
  details={'purpose':'Scale memory dimension to investigate capacity, using exact-query tasks.'}
  for s in seeds:
   for n in (64,256):
    for d in (64,128,256,512):rows.append(trial(s,n,d))
 else:raise ValueError(e)
 save(e,rows,details)

def run_more(e):
 rows=[]
 if e==5:
  for seed in range(100,112):
   for n in (64,128,256):
    for method in ('bind','linear','ridge','exact'):
     rows.append(trial(seed,n,128,method=method))
  save(e,rows,{'purpose':'Identical keys and labels; ridge is trained linear neural layer, not an untrained straw baseline. Ridge lambda=0.1 fixed, not tuned on reported memories.'})
 elif e==6:
  for seed in range(100,112):
   for sigma in (0.,0.3,0.8,1.5):
    for method,banks in [('bind',1),('bank',4),('bank',16),('linear',1),('ridge',1),('exact',1)]:
     rows.append(trial(seed,256,128,sigma=sigma,method=method,**({'banks':banks} if method=='bank' else {})))
  save(e,rows,{'purpose':'Corrupted phase queries, sigma in radians. Banks route using query content, no oracle item index. Same data/query per seed and sigma for all methods.'})
 elif e==7:
  for seed in range(100,112):
   for sigma in (0.,.3,.8):
    for method in ('bind','linear','ridge32'):
     for quant in (False,True):rows.append(trial(seed,256,128,sigma=sigma,method=method,quant=quant))
  save(e,rows,{'purpose':'Quantize persistent state to int8 plus a float32 scale. Codebook stays complex64. Arithmetic reconstructs float arrays: storage saving, not low-power inference evidence.'})
 else:raise ValueError(e)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('experiment',type=int);args=p.parse_args();(run if args.experiment<=4 else run_more)(args.experiment)
