"""57: fresh-trajectory check of a fixed confidence cap after noisy recalibration."""
from randomized_monitoring import *

def fresh_sequences():
 r=np.random.default_rng(201000);out={}
 for j in range(12):
  seq=np.zeros(24)
  if j%3:
   start,end=sorted(r.choice(np.arange(3,22),size=2,replace=False));level=float(r.uniform(.3,1.))
   if j%3==1:seq[start:end]=level
   else:seq[start:end]=np.linspace(0,level,end-start,endpoint=False);seq[end:]=np.linspace(level,0,24-end)
  out[str(j)]=seq.tolist()
 return out

def main():
 review();log('57_plan',{'reviewed':'56_result:triggeredaccuracy competitive,butabruptCE .289 worse thannever .277;XOR outliers dominate,alloptimizersconverged. Hypothesis:small noisy calibration sets permit extremeconfidence,not proven numericalfailure.','hypothesis':'A fixed prediction-confidence cap may reduce occasional extreme errors afterrecalibration without alteringlabels orquantummeasurement cost.','design':'12fresh stationary/pulse/ramp trajectories,new reference/calibration/testrandomness;12oldencoders. Same64calibrationlabels,24x32testpoints,512shotmonitor. Never,periodic6,triggered. Compare identicallogits raw andclipped[-4,4](probabilitiesabout1.8%..98.2%). Capfixedbeforetesting,notselectedontest. Same cap appliedtoallpolicies.','cost':'Identical quantumcost forraw/cappedpredictions. Sameinitial/inference/update/referencechargingas56.','scope':'A classical safeguard control,notquantumadvance;capchosenexploratorilyafter56mustbefurtherchecked. Noassuranceofcalibratedprobabilities orcoverage. Smallcalibrationoverfit remainsahypothesis untiltracesreviewed.'})
 sequences=fresh_sequences();schedules={k:monitor_schedule(v,202000+int(k)) for k,v in sequences.items()};rows=[];fits=[]
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task);C=np.random.default_rng(203000+s*100+tid).uniform(-np.pi,np.pi,(64,2));cy=target41(C,task)
  for key,seq in sequences.items():
   r=np.random.default_rng(204000+s*100000+tid*10000+int(key)*100);initial=affine_fit(learned_predict(C,base,sample52(interpolated_probs(C,base,0),base,r)),cy);names=('never','periodic6','triggered');current={k:initial for k in names};pred={k:[] for k in names};updates={k:0 for k in names};truth=[];trace=[]
   for t,f in enumerate(seq):
    fresh=affine_fit(learned_predict(C,base,sample52(interpolated_probs(C,base,f),base,r)),cy);trace.append(dict(batch=t,**fresh))
    for policy in names:
     if (policy=='triggered' and schedules[key]['alerts'][t]) or(policy=='periodic6' and t>0 and t%6==0):current[policy]=fresh;updates[policy]+=1
    T=r.uniform(-np.pi,np.pi,(32,2));truth.extend(target41(T,task));z=learned_predict(T,base,sample52(interpolated_probs(T,base,f),base,r))
    for policy in names:a=current[policy];pred[policy].extend(a['scale']*z+a['bias'])
   fits.append(dict(seed=s,task=task,trajectory=key,initial=initial,trace=trace))
   for policy in names:
    z=np.array(pred[policy]);reference=schedules[key]['reference_shots'] if policy=='triggered' else 0;cost=212992+reference+16384*updates[policy]
    assert np.array_equal(z>0,np.clip(z,-4,4)>0)
    for guard,zz in [('raw',z),('cap4',np.clip(z,-4,4))]:rows.append(dict(seed=s,task=task,trajectory=key,kind=('stationary','abrupt','ramp')[int(key)%3],policy=policy,guard=guard,updates=updates[policy],reference_shots=reference,total_shots=cost,max_abs_raw_logit=float(abs(z).max()),**metrics(zz,np.array(truth))))
  print(s,task,flush=True)
 log('57_result',{'rows':rows,'fits':fits,'sequences':sequences,'schedules':schedules,'decision':'Judgeprobabilityerror separatelyfromlabelsandcost. Fixedcapisnotacalibrationguarantee;next compareproperregularizedorheldoutcalibration withmoreindependentlabels andunseennoisemechanisms.'})
if __name__=='__main__':main()
