"""55: temporally ordered, charged monitoring-triggered affine recalibration."""
from reference_monitor import *
SEQUENCES={'stationary':[0.]*18,'abrupt':[0.]*6+[1.]*6+[0.]*6,'gradual':[0,0,0,.25,.25,.5,.5,.75,.75,1,1,1,.75,.5,.25,0,0,0]}
def interpolated_probs(X,base,fraction):
 a,b=ENV52['A'],ENV52['B'];e={k:(1-fraction)*np.array(a[k])+fraction*np.array(b[k]) for k in a};M=asymmetric_transition(e['q0'],e['q1']);P=basis_probabilities(X@np.array(base['input_matrix']),base['frequencies'],BASES)
 return [((1-e['p'])*p+e['p']/4)@M for p in P]
def monitor_schedule(seq,seed):
 r=np.random.default_rng(seed);baseline=reference_counts(reference_probabilities(0),4096,r);cutoff=threshold(baseline,512,r);alerts=[];scores=[]
 for f in seq:
  counts=reference_counts(reference_probabilities(f),512,r);score=float(statistic(baseline,counts));alert=score>cutoff;alerts.append(bool(alert));scores.append(score)
  if alert:baseline=reference_counts(reference_probabilities(f),4096,r);cutoff=threshold(baseline,512,r)
 return dict(alerts=alerts,scores=scores,reference_shots=4096+512*len(seq)+4096*sum(alerts))
def main():
 review();log('55_plan',{'reviewed':'54_result:512monitor shots gives.95% falsealarm and99.85% detection forfullA->B change;only9.5% forquarter and56.6% forhalf drift. Choose512beforepolicytest.','hypothesis':'Reference-triggered affine recalibration can retain most accuracy ofalwaysupdate while using fewer adaptation shots;small gradual changes may defeat it.','sequences':SEQUENCES,'design':'12frozen41 encoders,3fresh reference/noise trajectories persequence.18batches of64fresh test inputs.64fresh labeled calibration points/model,independent ofencoder train/validation andalltest data. Initial affinefit inA for allpolicies. Before each batch:neverupdate,alwaysrefreshaffine,or512shot monitor triggers refresh. Samebatchcalibration outcomes andtest measurements paired acrosspolicies.','adaptation':'Onalarm refresh4096shot reference baseline and64*256shot affinecalibration. No knowledge ofchannel fraction orfuture used bydetector. Reuse64known labels acrossupdates. Alwaysupdate doesnot runmonitor. Synchronous update beforebatch,no latency model.','cost':'Allpolicies:16384initialcalibration+294912inference shots/model/trajectory. Always:18*16384extra adaptation. Triggered:16384peralarm plus4096initialreference+18*512monitor+4096perreference refresh. Never:noextra. Chargeeachdeployment separately,noamortization acrossmodels.','scope':'Hypothetical commonchannels,prelistedsequences,3reference trajectories notindependentdevices;idealencoder reused,affine adaptationonly;noIBMhardware orquantumadvantage.'})
 schedules={};rows=[];fits=[]
 for i,(name,seq) in enumerate(SEQUENCES.items()):
  for rep in range(3):schedules[f'{name}:{rep}']=monitor_schedule(seq,181000+1000*i+rep)
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];taskid=TASKS41.index(task);C=np.random.default_rng(182000+s*100+taskid).uniform(-np.pi,np.pi,(64,2));cy=target41(C,task)
  for i,(name,seq) in enumerate(SEQUENCES.items()):
   for rep in range(3):
    schedule=schedules[f'{name}:{rep}'];r=np.random.default_rng(183000+s*100000+taskid*10000+i*1000+rep);F0=sample52(interpolated_probs(C,base,0),base,r);initial=affine_fit(learned_predict(C,base,F0),cy);current=initial;predictions={k:[] for k in ('never','always','triggered')};truth=[];updates=[]
    for t,f in enumerate(seq):
     Fcal=sample52(interpolated_probs(C,base,f),base,r);fresh=affine_fit(learned_predict(C,base,Fcal),cy);updates.append(dict(batch=t,success=fresh['success'],iterations=fresh['iterations']))
     if schedule['alerts'][t]:current=fresh
     T=r.uniform(-np.pi,np.pi,(64,2));y=target41(T,task);F=sample52(interpolated_probs(T,base,f),base,r);z=learned_predict(T,base,F);truth.extend(y)
     for policy,fit in [('never',initial),('always',fresh),('triggered',current)]:predictions[policy].extend(fit['scale']*z+fit['bias'])
    fits.append(dict(seed=s,task=task,sequence=name,repeat=rep,initial=initial,update_optimizer_status=updates))
    for policy,z in predictions.items():
     n=0 if policy=='never' else len(seq) if policy=='always' else sum(schedule['alerts']);monitorcost=schedule['reference_shots'] if policy=='triggered' else 0;adaptcost=n*64*256;rows.append(dict(seed=s,task=task,sequence=name,repeat=rep,policy=policy,updates=n,initial_calibration_shots=16384,inference_shots=294912,reference_shots=monitorcost,adaptation_shots=adaptcost,total_shots=16384+294912+monitorcost+adaptcost,**metrics(np.array(z),np.array(truth))))
  print(s,task,flush=True)
 log('55_result',{'rows':rows,'schedules':schedules,'fits':fits,'decision':'Review accuracy,CE andtotalcost together,including stationaryfalsealarms andgradualdelay. Do not claim monitoringworthwhile universally;next comparefixedperiodic updating atmatchedcost andvarylabel/reference budgets.'})
if __name__=='__main__':main()
