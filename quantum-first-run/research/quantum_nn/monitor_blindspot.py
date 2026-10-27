"""64: setting-dependent faults outside a reference panel's coverage."""
from temporal_budget import *

def basis_fault_probs(X,fit,bases,extra):
 e=ENV52['A'];M=asymmetric_transition(e['q0'],e['q1']);ideal=basis_probabilities(X@np.array(fit['input_matrix']),fit['frequencies'],bases);out=[]
 for i,P in enumerate(ideal):
  fault=extra if bases[i%len(bases)]=='YY' else 0.;p=1-(1-e['p'])*(1-fault);out.append(((1-p)*P+p/4)@M)
 return out

def main():
 review();log('64_plan',{
  'reviewed':'63_result: fresh-model monitoring does not consistently beat never/periodic policies;larger calibration budget improves probabilities. Previous monitor assumes reference and task settings share errors.',
  'hypothesis':'A ZZ/XX reference panel cannot detect a fault confined to YY measurements,even if predictions degrade. Adding YY may restore detection at the same monitoring budget.',
  'fault':'Extra40% global depolarization only in YY measurement settings,on top of fixedA channel. Other settings unchanged. This deliberately constructed setting-dependent stress test is not a measured IBM fault.',
  'design':'Compare4referencepoints with ZZ/XX versus ZZ/XX/YY,6144baseline shots and768/event for both. Threshold99th percentile bootstrap4000 using only baseline.1000fresh events,extraYYfault0/.4. Separately evaluate12fresh62 models,128independent labeled calibration examples measured inA once,2048testpoints,5measurement repeats per fault. Affine correction frozen across fault conditions.',
  'cost':'Same6144initialreference+768/event;expanded panel divides768over12groups rather than8. Prediction experiment:32768calibration+2048*256shots pertestrepeat. No monitor-triggered repair assumed.',
  'scope':'Coverage counterexample,not an unbiased survey of hardware fault frequencies. No automatic guarantee that adding one basis solves arbitrary circuit-dependent errors.'})
 monitor=[];rows=[];checks=[];identity=dict(input_matrix=np.eye(2),frequencies=[1.])
 for panel,bases in [('limited',['ZZ','XX']),('expanded',['ZZ','XX','YY'])]:
  P0=np.concatenate(basis_fault_probs(REFERENCES,identity,bases,0.));P1=np.concatenate(basis_fault_probs(REFERENCES,identity,bases,.4));difference=float(abs(P1-P0).max());checks.append(dict(panel=panel,probability_change=difference))
  if panel=='limited':assert difference==0.
  r=np.random.default_rng(np.random.SeedSequence([64,len(bases)]));baseline=reference_counts(P0,6144,r);cutoff=threshold(baseline,768,r)
  for fault,P in [(0.,P0),(.4,P1)]:
   counts=reference_counts(P,768,r,1000);scores=statistic(baseline,counts);monitor.append(dict(panel=panel,fault=fault,alert_rate=float(np.mean(scores>cutoff)),threshold=cutoff))
 for item in json.loads((LOG/'62_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task);r=np.random.default_rng(np.random.SeedSequence([64,s,tid]));C=r.uniform(-np.pi,np.pi,(128,2));cy=target41(C,task);Fc=sample52(basis_fault_probs(C,base,BASES,0),base,r);fit=affine_fit(learned_predict(C,base,Fc),cy);T=r.uniform(-np.pi,np.pi,(2048,2));y=target41(T,task)
  for fault in (0.,.4):
   P=basis_fault_probs(T,base,BASES,fault)
   for rep in range(5):
    F=sample52(P,base,np.random.default_rng(np.random.SeedSequence([64,s,tid,rep,1])));z=cal_predict(learned_predict(T,base,F),fit);rows.append(dict(seed=s,task=task,fault=fault,repeat=rep,**metrics(z,y)))
 log('64_result',{'monitor':monitor,'rows':rows,'coverage_checks':checks,'decision':'Reference coverage must be specified and challenged. Do not claim universal detection or recommend hardware monitoring from shared-channel toy success;next prioritize robust independent calibration and circuit-representative panels.'})
if __name__=='__main__':main()
