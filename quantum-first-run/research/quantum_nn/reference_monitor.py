"""54: unlabeled reference-circuit monitor with bootstrap null threshold."""
from asymmetric_drift import *
REFERENCES=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
REF_BASES=['ZZ','XX']

def reference_probabilities(fraction):
 a,b=ENV52['A'],ENV52['B'];e={k:(1-fraction)*np.array(a[k])+fraction*np.array(b[k]) for k in a};P=basis_probabilities(REFERENCES,[1.],REF_BASES);M=asymmetric_transition(e['q0'],e['q1'])
 return np.concatenate([((1-e['p'])*p+e['p']/4)@M for p in P])
def reference_counts(P,budget,r,repeats=None):
 assert budget%len(P)==0
 n=budget//len(P)
 return np.stack([r.multinomial(n,p,size=repeats) if repeats is not None else r.multinomial(n,p) for p in P],axis=-2)
def statistic(A,B):
 na=A.sum(-1,keepdims=True);nb=B.sum(-1,keepdims=True);pooled=(A+B)/(na+nb);ea=pooled*na;eb=pooled*nb
 def term(c,e):return np.where(c>0,c*np.log(np.maximum(c,1e-300)/np.maximum(e,1e-300)),0.)
 return 2*(term(A,ea)+term(B,eb)).sum(axis=(-1,-2))
def threshold(baseline,budget,r,draws=4000):
 P=(baseline+.5)/(baseline.sum(-1,keepdims=True)+2)
 A=reference_counts(P,int(baseline.sum()),r,draws);B=reference_counts(P,budget,r,draws)
 return float(np.quantile(statistic(A,B),.99,method='higher'))
def main():
 review();log('54_plan',{'reviewed':'53_checkpoint and52_result: matched correction helps,but stale calibration can hurt;no monitor exists yet.','hypothesis':'A small unlabeled reference-circuit panel can flag channel changes without knowing B or using task labels.','references':REFERENCES,'bases':REF_BASES,'design':'Fixed identity-encoder reference circuits;8 input/basis groups.4096initial baseline shots,monitor budgets128/512/2048 total per event. Compare joint categorical outcome counts with two-sample likelihood-ratio statistic.99th-percentile threshold from4000 parametric bootstrap pairs generated from smoothed baseline only. Detector never receives B probabilities.2000 fresh events per budget/fraction0,.25,.5,1.','cost':'4096initial quantum shots+budget per monitoring event. Bootstrap draws are classical calculations,not quantum executions. Evaluation repeats are stationary synthetic monitoring events,not independent devices or drift trajectories.','scope':'Shared homogeneous hypothetical A/B channels;reference and task circuits assumed affected identically. Unknown intermediate fractions used only for evaluation;no trained-model decision orrecalibration yet. Nominal1% falsealarm is tested,not guaranteed.'})
 r=np.random.default_rng(171000);baseline=reference_counts(reference_probabilities(0),4096,r);rows=[]
 for budget in (128,512,2048):
  cutoff=threshold(baseline,budget,np.random.default_rng(172000+budget))
  for j,fraction in enumerate((0.,.25,.5,1.)):
   counts=reference_counts(reference_probabilities(fraction),budget,np.random.default_rng(173000+budget+10000*j),2000);scores=statistic(baseline,counts);rows.append(dict(budget=budget,fraction=fraction,threshold=cutoff,alert_rate=float(np.mean(scores>cutoff)),alerts=scores>cutoff,scores=scores))
 log('54_result',{'rows':rows,'baseline_counts':baseline,'decision':'Review actual falsealarm and detection rates before picking a monitoring budget. Then test a charged end-to-end recalibration policy on temporal channel sequences;detection alone does not establish net benefit.'})
if __name__=='__main__':main()
