"""Later experiments, chosen after inspecting earlier outputs."""
import time
import numpy as np
from lab import data,scores,int8_state,save,RESULTS

def streaming():
 rows=[];errors=[]
 for seed in range(100,112):
  n=128;d=128;K,V,y,r=data(seed,n,d);c=len(V);initial=y.copy()
  Y=np.eye(c,dtype=np.float32)[y];original=K.conj().T@Y/d
  states={m:original.copy() for m in ('append','decay','correct','correct_int8')}
  for _ in range(256):
   i=int(r.integers(n));old=int(y[i]);new=int((old+r.integers(1,c))%c);y[i]=new
   inc=np.zeros(c,dtype=np.float32);inc[new]=1;delta=inc.copy();delta[old]-=1
   add=np.outer(K[i].conj(),inc)/d;change=np.outer(K[i].conj(),delta)/d
   states['append']+=add;states['decay']=.98*states['decay']+add
   states['correct']+=change
   states['correct_int8'],_=int8_state(states['correct_int8']+change)
  rebuilt=K.conj().T@np.eye(c,dtype=np.float32)[y]/d
  err=float(np.max(abs(states['correct']-rebuilt)));errors.append(err);assert err<1e-5
  for m,state in states.items():
   payload=4100 if m=='correct_int8' else state.nbytes
   metadata=4*n if m.startswith('correct') else 0
   for scope,mask in [('all',np.ones(n,dtype=bool)),('changed',y!=initial)]:
    S=(K@state).real
    rows.append({'seed':seed,'n':n,'d':d,'method':m,'scope':scope,'accuracy':float(np.mean(S.argmax(1)[mask]==y[mask])),'state_bytes':int(payload),'metadata_bytes':metadata,'total_persistent_bytes':int(payload+metadata)})
 save(8,rows,{'updates':256,'decay_per_update':.98,'max_incremental_rebuild_error':max(errors),'contract':'Each update supplies a stable slot ID, the full key and new label. Corrected methods retain old labels (512 bytes); key generation/storage outside the supplied input is not charged. No online learning/generalization claim.'})

if __name__=='__main__':streaming()
