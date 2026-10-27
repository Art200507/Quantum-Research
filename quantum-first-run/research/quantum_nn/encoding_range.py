"""Experiment 11: encoding range versus measurement-resolution tradeoff."""
from core import *
from robustness import halving

def main():
 review();previous=json.loads((LOG/'10_result.json').read_text())
 log('11_plan',{'hypothesis':'Smaller data-encoding frequencies may improve exact ranking under shifted input ranges while shrinking measurable probability gaps.','choice_origin':'Frequency grid1/.5/.25/.125 chosen after seeing scale4 failure in experiment10; this is an exploratory intervention, not pre-existing confirmation.','design':'Freeze five circuit fits; new 128-episode datasets,16 candidates; input scales1/2/4; 512 and8192 total shots/query;10 repeats. Evaluate full grid, not a selected winner.','scope':'No train/test-informed retraining or oracle range normalization; external frequency multiplier applies equally to query and key. Sampling from verified ideal two-qubit Born probabilities.','reviewed_previous_rows':len(previous['rows'])})
 rows=[];fits=json.loads((LOG/'04_result.json').read_text())['fits']
 for fit in fits:
  seed=fit['seed'];p=np.array(fit['parameters']);Q,K,y=episodes(35000+seed,128,16)
  for scale in (1.,2.,4.):
   for frequency in (1.,.5,.25,.125):
    P=overlaps(Q*scale*frequency,K*scale*frequency,p[:4],reupload_scaled=True)
    ordered=np.sort(P,axis=1);gap=ordered[:,-1]-ordered[:,-2]
    rows.append({'seed':seed,'scale':scale,'frequency':frequency,'method':'quantum_exact','budget':0,'repeat':0,'accuracy':float(np.mean(P.argmax(1)==y)),'mean_top_gap':float(np.mean(gap))})
    for B in (512,8192):
     for rep in range(10):
      r=np.random.default_rng(36000+seed*10000+B+rep)
      pred,used,rounds=halving(P,B,r)
      rows.append({'seed':seed,'scale':scale,'frequency':frequency,'method':'adaptive','budget':used,'rounds':rounds,'repeat':rep,'accuracy':float(np.mean(pred==y))})
 log('11_result',{'rows':rows,'decision':'Separate representational ranking from shot-limited distinguishability. Inspect all conditions; a good ideal score is insufficient for a useful measured quantum head.'})

if __name__=='__main__':main()
