"""63: paired calibration budgets during drift, using fresh trained encoders."""
from fresh_encoders import *

def new_trajectories():
 r=np.random.default_rng(np.random.SeedSequence([63,0]));out={}
 for j in range(6):
  seq=np.zeros(24)
  if j%3:
   start,end=sorted(r.choice(np.arange(3,22),2,replace=False));level=float(r.uniform(.3,1.))
   if j%3==1:seq[start:end]=level
   else:seq[start:end]=np.linspace(0,level,end-start,endpoint=False);seq[end:]=np.linspace(level,0,24-end)
  out[str(j)]=seq.tolist()
 return out

def main():
 review();log('63_plan',{
  'reviewed':'62_result: all12 fresh encoders retained;26/48 optimizer runs converged,Qiskit max error3.33e-16.61/60: larger calibration budgets help static mean error;temporal behavior unconfirmed.',
  'hypothesis':'More calibration labels may stabilize temporal correction,but adaptation cost may erase monitoring advantage.',
  'design':'12fresh62 models,6new24batch trajectories(two each stationary,pulse,ramp),32fresh test inputs/batch.256new labeled calibration points/model/trajectory independent encoder training/selection/test. Nested64/256 prefixes;256shots/point. All methods/budgets share calibration/test measurement records and monitoring schedules.',
  'policies':'Never,periodic6,always,512shot triggered. All initial fits precede batch0. Always refreshes only batches1..23,removing duplicate initial refresh. Same affineMLE fitting;raw/cap4 predictions both retained,cap not selected on test.',
  'cost':'Inference196608shots;initialN*256;N*256 per policy update. Triggered reference4096+24*512+4096peralarm. Equal methods within budget;different budgets explicitly costed. No pooling/amortization across model deployments.',
  'scope':'Fresh encoder seeds improve independence,but only3newtrainingseeds/task and2trajectories/kind. Same homogeneous hypothetical channel axis;not hardware or unseen noise mechanisms.'})
 sequences=new_trajectories();schedules={k:monitor_schedule(v,240000+int(k)) for k,v in sequences.items()};rows=[];fits=[]
 for item in json.loads((LOG/'62_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task)
  for key,seq in sequences.items():
   r=np.random.default_rng(np.random.SeedSequence([63,s,tid,int(key)]));C=r.uniform(-np.pi,np.pi,(256,2));cy=target41(C,task);F0=sample52(interpolated_probs(C,base,0),base,r);z0=learned_predict(C,base,F0);names=('never','always','periodic6','triggered');current={};pred={};updates={};trace=[];truth=[]
   for N in (64,256):
    initial=affine_fit(z0[:N],cy[:N]);current[N]={k:initial for k in names};pred[N]={k:[] for k in names};updates[N]={k:0 for k in names};trace.append(dict(batch=-1,N=N,**initial))
   for t,f in enumerate(seq):
    zc=learned_predict(C,base,sample52(interpolated_probs(C,base,f),base,r));T=r.uniform(-np.pi,np.pi,(32,2));truth.extend(target41(T,task));z=learned_predict(T,base,sample52(interpolated_probs(T,base,f),base,r))
    for N in (64,256):
     fresh=affine_fit(zc[:N],cy[:N]);trace.append(dict(batch=t,N=N,**fresh))
     for policy in names:
      refresh=(policy=='always' and t>0) or(policy=='periodic6' and t>0 and t%6==0) or(policy=='triggered' and schedules[key]['alerts'][t])
      if refresh:current[N][policy]=fresh;updates[N][policy]+=1
      fit=current[N][policy];pred[N][policy].extend(fit['scale']*z+fit['bias'])
   fits.append(dict(seed=s,task=task,trajectory=key,trace=trace))
   for N in (64,256):
    for policy in names:
     z=np.array(pred[N][policy]);reference=schedules[key]['reference_shots'] if policy=='triggered' else 0;count=updates[N][policy];total=196608+N*256*(1+count)+reference
     assert np.array_equal(z>0,np.clip(z,-4,4)>0)
     for guard,zz in [('raw',z),('cap4',np.clip(z,-4,4))]:rows.append(dict(seed=s,task=task,trajectory=key,kind=('stationary','pulse','ramp')[int(key)%3],calibration_labels=N,policy=policy,guard=guard,updates=count,reference_shots=reference,initial_calibration_shots=N*256,adaptation_shots=N*256*count,inference_shots=196608,total_shots=total,**metrics(zz,np.array(truth))))
  print(s,task,flush=True)
 log('63_result',{'rows':rows,'fits':fits,'sequences':sequences,'schedules':schedules,'decision':'Review probability quality and cost jointly for both calibration budgets. Confirm any new pattern before choosing deployment defaults;next challenge circuit-dependent errors invisible to reference panels.'})
if __name__=='__main__':main()
