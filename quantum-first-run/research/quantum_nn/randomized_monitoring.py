"""56: unseen drift times/magnitudes and fixed periodic-update controls."""
from triggered_recalibration import *

def make_sequences():
 r=np.random.default_rng(191000);out={}
 for j in range(12):
  seq=np.zeros(24)
  if j%3:
   times=sorted(r.choice(np.arange(3,22),size=2,replace=False));level=float(r.uniform(.3,1.))
   if j%3==1:seq[times[0]:times[1]]=level
   else:
    seq[times[0]:times[1]]=np.linspace(0,level,times[1]-times[0],endpoint=False);seq[times[1]:]=np.linspace(level,0,24-times[1])
  out[str(j)]=seq.tolist()
 return out

def main():
 review();log('56_plan',{'reviewed':'55_result:abrupt triggered90.92% accuracy vsalways90.91%,365568vs606208shots;gradual small accuracy loss;stationary monitorpureoverhead. Missingperiodic comparator andnewtimings.','hypothesis':'Triggered recalibration provides a better accuracy/cost tradeoff than fixed periodicupdates underunseenchange timings,butneednotdominatecheap schedules.','design':'12new24batch device trajectories:4stationary,4random abruptpulses,4random ramps. Change times/magnitudes not supplied todetector orpolicies.12frozen encoders,32fresh testinputs/batch,64fresh knowncalibrationlabels/model independenttrain/validation/test. Policiesnever,always,periodic3/6/8,and512shot monitor. Periodic updates beforebatches3/6/etc,excluding0(initialfitshared). Allmethods useidentical batch measurement/calibration outcomes.','cost':'196608inference+16384initialcalibration shots;16384perupdate. Monitoralso4096initialreference+24*512checks+4096peralarmbaseline refresh. No cost-sharing acrosspolicies assumed.','selection':'All schedules/budgets frozenbeforetest;nooracle changepointupdates orchoosingbestperiodontest.','scope':'Unseen times/magnitudes onpreviousA->Bchannelaxis,not unseen noise mechanism.12device trajectories sharedacrossmodels;3oldtrainingseeds/task. NoIBMdevice claim.'})
 sequences=make_sequences();schedules={k:monitor_schedule(v,192000+int(k)) for k,v in sequences.items()};rows=[];fits=[]
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task);C=np.random.default_rng(193000+s*100+tid).uniform(-np.pi,np.pi,(64,2));cy=target41(C,task)
  for key,seq in sequences.items():
   r=np.random.default_rng(194000+s*100000+tid*10000+int(key)*100);F0=sample52(interpolated_probs(C,base,0),base,r);initial=affine_fit(learned_predict(C,base,F0),cy);names=('never','always','periodic3','periodic6','periodic8','triggered');current={k:initial for k in names};pred={k:[] for k in names};updates={k:0 for k in names};truth=[];statuses=[]
   for t,f in enumerate(seq):
    Fcal=sample52(interpolated_probs(C,base,f),base,r);fresh=affine_fit(learned_predict(C,base,Fcal),cy);statuses.append(fresh['success'])
    for policy in names:
     trigger=policy=='always' or (policy=='triggered' and schedules[key]['alerts'][t]) or (policy.startswith('periodic') and t>0 and t%int(policy[8:])==0)
     if trigger:current[policy]=fresh;updates[policy]+=1
    T=r.uniform(-np.pi,np.pi,(32,2));y=target41(T,task);Ft=sample52(interpolated_probs(T,base,f),base,r);z=learned_predict(T,base,Ft);truth.extend(y)
    for policy in names:a=current[policy];pred[policy].extend(a['scale']*z+a['bias'])
   fits.append(dict(seed=s,task=task,trajectory=key,initial_success=initial['success'],update_successes=statuses))
   for policy in names:
    reference=schedules[key]['reference_shots'] if policy=='triggered' else 0;adaptation=updates[policy]*16384;rows.append(dict(seed=s,task=task,trajectory=key,kind=('stationary','abrupt','ramp')[int(key)%3],policy=policy,updates=updates[policy],reference_shots=reference,adaptation_shots=adaptation,total_shots=196608+16384+reference+adaptation,**metrics(np.array(pred[policy]),np.array(truth))))
  print(s,task,flush=True)
 log('56_result',{'rows':rows,'sequences':sequences,'schedules':schedules,'fits':fits,'decision':'Reviewperiodic schedules alongsideadaptivepolicy as acost/accuracyfrontier. Do notclaimmonitoringwinsifperiodic achievescomparableaccuracycheaply;nextvarycalibrationbudget andtestunseennoise mechanisms.'})
if __name__=='__main__':main()
