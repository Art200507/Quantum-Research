"""Experiment14: structural checks and a concrete encoding collision."""
from core import *
from hybrid_block import dataset,forward
from robustness import halving
from shots import adaptive

def main():
 review();log('14_plan',{'hypothesis':'Confirm an exact periodic encoding collision rather than attributing every range-shift failure to sampling. Check hybrid-block candidate permutation invariance and generalized allocator accounting.','purpose':'Validation and failure isolation; no new training or novelty claim.'})
 fits=json.loads((LOG/'12_result.json').read_text())['fits'];checks=[];collisions=[]
 for fit in fits:
  p=np.array(fit['parameters']);seed=fit['seed'];kind=fit['kind'];Q,K,v,y=dataset(53000+seed,16)
  perm=np.random.default_rng(54000+seed).permutation(8)
  error=float(np.max(abs(forward((Q,K,v,y),p,kind)-forward((Q,K[:,perm],v[:,perm],y),p,kind))))
  assert error<1e-10;checks.append({'seed':seed,'kind':kind,'candidate_permutation_error':error})
  if kind=='quantum':
   q=np.zeros((1,2));far=np.array([2*np.pi/p[0],0.]);keys=np.stack([q[0],far])[None,:,:]
   P=overlaps(q,keys,p[:4],reupload_scaled=True)
   qc=circuit(far,p[:4],True,reupload_scaled=True);qc.compose(circuit(q[0],p[:4],True,reupload_scaled=True).inverse(),inplace=True)
   born=float(abs(Statevector.from_instruction(qc).data[0])**2)
   assert np.allclose(P,1,atol=1e-12) and abs(born-1)<1e-12
   collisions.append({'seed':seed,'query':q[0],'far_key':far,'target_squared_distance':float(far[0]**2),'kernel_probabilities':P[0],'Qiskit_far_overlap':born})
 for m in (4,8,16,32):
  P=np.random.default_rng(m).uniform(size=(8,m));pred,used,rounds=halving(P,512,np.random.default_rng(1));assert len(pred)==8 and used<=512 and np.all(pred<m)
 log('14_result',{'permutation_checks':checks,'exact_periodic_collisions':collisions,'decision':'Encoding maps a nonzero-distance key to exactly the same state as query. More shots cannot resolve this constructed case. Future model must control input range or change feature map; permutation invariance and budgets pass.'})

if __name__=='__main__':main()
