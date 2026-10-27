"""Trainable quantum overlap attention and architecture diagnostics."""
from core import *
from qiskit import transpile

def diagnostic():
 review();log('02_plan',{'hypothesis':'Shared trainable gates after all data encoding cannot change fidelity attention; check before wasting training shots. Also check the QNN local-observable parameter null direction.'})
 Q,K,y=episodes(4000,32);p=np.array([1.,1.,.2,-.3]);base=overlaps(Q,K,p,True)
 diffs=[];grads=[];inter=[]
 for j in (2,3):
  other=p.copy();other[j]+=1.1;diffs.append(float(np.max(abs(overlaps(Q,K,other,True)-base))))
  other=p.copy();other[j]+=1e-4;grads.append(float(np.linalg.norm((overlaps(Q,K,other,True)-base)/1e-4)))
  inter.append(float(np.max(abs(overlaps(Q,K,other,False)-overlaps(Q,K,p,False)))/1e-4))
 X=np.random.default_rng(4).normal(size=(32,2));null=float(np.max(abs(qnn_z(X,[.1,.2])-qnn_z(X,[1.7,.2]))))
 assert max(diffs)<1e-12 and max(grads)<1e-8 and null<1e-12
 # Independently verify a compute-uncompute kernel circuit against the overlap.
 qc=circuit(K[0,0],p,True);qc.compose(circuit(Q[0],p,True).inverse(),inplace=True)
 prob=float(abs(Statevector.from_instruction(qc).data[0])**2)
 err=abs(prob-overlaps(Q[:1],K[:1,:1],p)[0,0]);assert err<1e-12
 isa=transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=7)
 log('02_result',{'shared_tail_max_changes':diffs,'shared_tail_gradient_norms':grads,'interleaved_max_gradient_components':inter,'QNN_unused_parameter_change':null,'compute_uncompute_probability_error':err,'kernel_cx':int(isa.count_ops().get('cx',0)),'kernel_depth':isa.depth(),'explanation':'<U phi(q)|U phi(k)> = <phi(q)|phi(k)>; a common final unitary cancels. Local QNN rotation on unmeasured qubit also does not change this observable.','decision':'Use interleaved data encoding for trainable attention; remove unobservable parameters in later QNNs. This is known algebra, now verified in our implementation.'})

def train_attention(reupload_scaled=False):
 review();stage='04' if reupload_scaled else '03';log(stage+'_plan',{'revision':'Scale every re-upload consistently; earlier unscaled RZ keeps injecting the irrelevant coordinate.' if reupload_scaled else 'Initial interleaved architecture','hypothesis':'Interleaving data gates with trainable gates permits learning a task-specific quantum similarity. Test disjoint retrieval episodes, not training-set memorization.','task':'8 candidate keys and query with 2 real features. Correct candidate minimizes dx0^2 + .05 dx1^2. Values are one-hot positions, weighted by softmax(beta*fidelity).','budget':'5 training seeds, 128 train/128 validation/512 test episodes; max100 iterations; no noisy hardware optimization.'})
 rows=[];fits=[]
 for seed in range(5):
  offset=10000 if reupload_scaled else 0
  train=episodes(4100+seed+offset,128);valid=episodes(4200+seed+offset,128);test=episodes(4300+seed+offset,512)
  def loss(p,ds=train):
   Q,K,y=ds;z=np.exp(p[4])*overlaps(Q,K,p[:4],reupload_scaled=reupload_scaled);return float(np.mean(np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y]))
  init=np.array([1.,1.,.3,-.3,np.log(10.)]);start=time.perf_counter()
  res=minimize(loss,init,method='L-BFGS-B',bounds=[(.02,2.),(.02,2.),(-np.pi,np.pi),(-np.pi,np.pi),(0,np.log(80.))],options={'maxiter':100,'ftol':1e-9})
  best=res.x
  # Classical metric attention learns relative coordinate weighting, not given oracle weight.
  def closs(p):
   Q,K,y=train;delta=(K-Q[:,None,:])**2;z=-np.exp(p[1])*(delta[:,:,0]+np.exp(p[0])*delta[:,:,1]);return np.mean(np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y])
  cres=minimize(closs,[0.,np.log(10.)],method='L-BFGS-B',bounds=[(-7,3),(0,np.log(80))],options={'maxiter':100})
  fits.append({'seed':seed,'parameters':best,'success':bool(res.success),'message':str(res.message),'iterations':res.nit,'calls':res.nfev,'seconds':time.perf_counter()-start,'classical_parameters':cres.x,'classical_success':bool(cres.success)})
  for split,ds in [('validation',valid),('test',test)]:
   Q,K,y=ds;delta=(K-Q[:,None,:])**2
   methods={'quantum_fixed':overlaps(Q,K,init[:4],reupload_scaled=reupload_scaled),'quantum_trained':overlaps(Q,K,best[:4],reupload_scaled=reupload_scaled),'classical_dot':np.einsum('ni,nmi->nm',Q,K),'classical_euclidean':-delta.sum(2),'classical_trained_metric':-(delta[:,:,0]+np.exp(cres.x[0])*delta[:,:,1]),'oracle_task_metric':-(delta[:,:,0]+.05*delta[:,:,1])}
   for name,S in methods.items():rows.append({'seed':seed,'split':split,'model':name,'accuracy':float(np.mean(S.argmax(1)==y))})
 log(stage+'_result',{'rows':rows,'fits':fits,'decision':'Inspect learned versus fixed kernel and trained classical metric; freeze parameters before shot-budget experiments. This is a quantum attention head with classical softmax/value aggregation, not a full transformer.'})

if __name__=='__main__':
 import sys
 diagnostic() if sys.argv[1]=='2' else train_attention(reupload_scaled=sys.argv[1]=='4')
