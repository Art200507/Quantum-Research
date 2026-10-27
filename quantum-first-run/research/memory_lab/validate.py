"""Experiment 10: frozen methods on fresh random memories; payload/timing ledger."""
import time
import numpy as np
from scipy.linalg import solve
from lab import data,noisy,save

def pack(A):
 if np.iscomplexobj(A):
  scale=np.float32(max(abs(A.real).max(),abs(A.imag).max())/127 or 1)
  a=np.round(A.real/scale).astype(np.int8);b=np.round(A.imag/scale).astype(np.int8)
  return lambda:(a.astype(np.float32)+1j*b.astype(np.float32))*scale,a.nbytes+b.nbytes+4
 scale=np.float32(abs(A).max()/127 or 1);a=np.round(A/scale).astype(np.int8)
 return lambda:a.astype(np.float32)*scale,a.nbytes+4

def build(K,V,y,method):
 n,d=K.shape;c=len(V)
 if method=='bind':
  A=np.sum(K*V[y],axis=0)
  return lambda Q:((Q.conj()*A)@V.conj().T).real/d,A.nbytes+V.nbytes
 if method in ('exact','exact8'):
  labels=y.astype(np.int32).copy()
  if method=='exact8':decode,stored=pack(K)
  else:
   A=K.copy();decode=lambda:A;stored=A.nbytes
  # Prediction directly outputs class IDs; no unnecessary one-hot workspace.
  return lambda Q:labels[((Q@decode().conj().T).real).argmax(1)],stored+labels.nbytes
 if method in ('linear','linear8'):
  A=K.conj().T@np.eye(c,dtype=np.float32)[y]/d
  if method=='linear8':decode,stored=pack(A)
  else:decode=lambda:A;stored=A.nbytes
  return lambda Q:(Q@decode()).real,stored
 if method in ('ridge32','ridge8'):
  X=np.concatenate([K.real,K.imag],1)/np.float32(np.sqrt(d))
  A=solve(X.T@X+np.float32(.1)*np.eye(2*d,dtype=np.float32),X.T@np.eye(c,dtype=np.float32)[y],assume_a='pos')
  assert A.dtype==np.float32
  if method=='ridge8':decode,stored=pack(A)
  else:decode=lambda:A;stored=A.nbytes
  return lambda Q:(np.concatenate([Q.real,Q.imag],1)/np.float32(np.sqrt(d)))@decode(),stored
 raise ValueError(method)

def main():
 rows=[];methods=('bind','linear','linear8','ridge32','ridge8','exact','exact8')
 for seed in range(1000,1024):
  for n in (128,256,512):
   K,V,y,r=data(seed,n,128)
   # Fresh corruption from deterministic per-condition generator for paired queries.
   queries={s:noisy(K,np.random.default_rng(seed+int(s*10000)),s) if s else K for s in (0.,.3,.8)}
   for method in methods:
    start=time.perf_counter();predict,stored=build(K,V,y,method);build_ms=(time.perf_counter()-start)*1000
    for sigma,Q in queries.items():
     result=predict(Q);pred=result if result.ndim==1 else result.argmax(1)
     row={'seed':seed,'n':n,'d':128,'sigma':sigma,'method':method,'accuracy':float(np.mean(pred==y)),'total_persistent_bytes':int(stored),'build_ms':build_ms}
     # Timing only one seed: warmup, then 25 repeated full batches. Includes dequantization.
     if seed==1000:
      durations=[]
      for _ in range(3):predict(Q)
      for _ in range(25):
       start=time.perf_counter();predict(Q);durations.append(time.perf_counter()-start)
      row['batch_query_us_median']=float(np.median(durations)*1e6)
      row['batch_query_us_iqr']=float((np.quantile(durations,.75)-np.quantile(durations,.25))*1e6)
     rows.append(row)
 # Timing/build metrics should not fragment scientific-condition summaries.
 clean=[{k:v for k,v in row.items() if k not in ('build_ms','batch_query_us_median','batch_query_us_iqr')} for row in rows]
 timings=[{k:v for k,v in row.items() if k!='accuracy'} for row in rows if row['seed']==1000]
 save(10,clean,{'timings_seed_1000':timings,'seeds':'1000..1023 fresh','ridge_lambda':.1,'all_arrays':'complex64/float32 except packed int8','timing':'25 warmed full-batch repeats, one seed, includes dequantization; diagnostic not device-independent speed evidence','excluded':'Training examples/build workspace, input encoding, Python object overhead, process RSS, battery/energy, full transformer, learned key encoder.'})

if __name__=='__main__':main()
