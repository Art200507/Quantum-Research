"""Experiment 10 (after checkpoint 09): candidate scaling and domain shift."""
from core import *
from shots import choose,adaptive

def halving(P,B,r):
 n,m=P.shape
 if m<2 or m&(m-1):raise ValueError('power of two candidates required')
 rounds=int(np.log2(m));wins=np.zeros_like(P);shots=np.zeros_like(P);active=np.tile(np.arange(m),(n,1));used=0
 for stage in range(rounds):
  count=active.shape[1];allocation=(B-used if stage==rounds-1 else B//rounds)//count
  if allocation<1:raise ValueError('insufficient budget')
  row=np.arange(n)[:,None];wins[row,active]+=r.binomial(allocation,np.take_along_axis(P,active,1));shots[row,active]+=allocation
  used+=count*allocation;estimate=wins[row,active]/shots[row,active]+r.uniform(0,1e-10,active.shape)
  if stage==rounds-1:pred=active[np.arange(n),estimate.argmax(1)]
  else:active=np.take_along_axis(active,np.argsort(estimate,1)[:,-count//2:],1)
 assert used<=B
 return pred,used,rounds

def main():
 # Read substantive checkpoint and latest results, not provenance metadata.
 review();checkpoint=json.loads((LOG/'09_checkpoint.json').read_text())
 log('10_plan',{'hypothesis':'Adaptive readout gains may survive more candidates but periodic quantum encoding may fail under input-domain shift. Freeze maps and classical metrics from experiment4.','design':'5 trained seeds x256 fresh episodes, m=4/8/16/32, input multiplier1/2/4, B512 shots/query,20 repeats. Learned target metric remains dx0²+.05dx1². Common scaling leaves correct indices unchanged.','checks':'Generalized halving must reproduce old allocator for eight candidates with identical randomness. Costs include all screening shots; control rounds reported, not assumed free.','previous_checkpoint_read':checkpoint['focus']})
 P=np.random.default_rng(1).uniform(size=(20,8))
 a,used=adaptive(P,512,np.random.default_rng(2));b,used2,_=halving(P,512,np.random.default_rng(2));assert np.array_equal(a,b) and used==used2
 fits=json.loads((LOG/'04_result.json').read_text())['fits'];rows=[]
 for f in fits:
  seed=f['seed'];p=np.array(f['parameters']);w=np.exp(f['classical_parameters'][0])
  for m in (4,8,16,32):
   for scale in (1.,2.,4.):
    Q,K,y=episodes(31000+seed*100+m,256,m);Q=Q*scale;K=K*scale
    P=overlaps(Q,K,p[:4],reupload_scaled=True);delta=(K-Q[:,None,:])**2
    for method,S in [('quantum_exact',P),('classical_learned_metric',-(delta[:,:,0]+w*delta[:,:,1]))]:
     rows.append({'seed':seed,'m':m,'scale':scale,'method':method,'repeat':0,'accuracy':float(np.mean(S.argmax(1)==y)),'budget':0,'rounds':0})
    for rep in range(20):
     r=np.random.default_rng(32000+seed*10000+m*100+rep)
     uniform=choose(r.binomial(512//m,P)/(512//m),r)
     pred,used,rounds=halving(P,512,r)
     for method,out,shots,rounds_used in [('uniform',uniform,512,1),('adaptive',pred,used,rounds)]:
      rows.append({'seed':seed,'m':m,'scale':scale,'method':method,'repeat':rep,'accuracy':float(np.mean(out==y)),'budget':shots,'rounds':rounds_used})
 log('10_result',{'rows':rows,'checks':['eight-candidate legacy identity','power-of-two candidates','budget bound'],'decision':'Inspect ideal-versus-sampled collapse before choosing next intervention. Candidate scaling and distribution shift are distinct; do not attribute encoding aliasing to shot noise.'})

if __name__=='__main__':main()
