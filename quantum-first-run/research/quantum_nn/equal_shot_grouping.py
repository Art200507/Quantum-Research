"""48: regroup a fixed measurement record, without additional train shots."""
from confidence_control import *

def main():
 review();log('48_plan',{'reviewed':'47_result: noise readouts outperform ideal under toy channels but all degrade;device noise does not establish equal training cost.','hypothesis':'Regrouping one fixed shot record can match low-shot retraining with no additional quantum measurement cost.','design':'Frozen45 five encoders at256shot inference. Reconstruct original four independent256shot training blocks. Group4 blocks into1 averaged1024shot feature,2 averaged512shot features,or4 separate256shot features per labeled point. Identical total1024training shots per point; original fixed noisy validation draws select lambda and grouping by CE. Fresh2048test points,10 paired256shot repeats.','controls':'G1,G2,G4 andvalidation-selectedG;reuse45 temperature/ideal only for context if needed. No new training labels.','source':'https://arxiv.org/abs/2604.28160 (abstract read): fixed-shot grouping and readout regularization already studied;this is an independent toy control,not a replication of its full implementation or a novel method.','scope':'Ideal encoder and sampling noise only;128 unique training labels regardless of grouping. Validation selection incurs classical optimization and no extra quantum shots across groups.'})
 old=json.loads((LOG/'45_result.json').read_text());rows=[];fits=[];checks=[]
 for item in old['fits']:
  if item['budget']!=256:continue
  s,base=item['seed'],item['base'];X=np.random.default_rng(141000+s).uniform(-np.pi,np.pi,(384,2));y=target41(X,'annulus');r=np.random.default_rng(143000+s*10000+256);blocks=np.stack([noisy_features(X[:128],base,256,r) for _ in range(4)]);V=np.array(item['validation_features']);vy=np.array(item['validation_labels']);candidates=[]
  for G in (1,2,4):
   F=blocks.reshape(G,4//G,128,-1).mean(1).reshape(G*128,-1);fit=refit(F,np.tile(y[:128],G),V,vy,base);candidates.append(dict(groups=G,fit=fit))
  g4=next(f['fit'] for f in candidates if f['groups']==4);err=float(np.max(abs(np.array(g4['weights'])-np.array(item['noise_fit']['weights']))));assert err<1e-9;checks.append(err)
  selected=min(candidates,key=lambda c:c['fit']['validation_loss']);fits.append(dict(seed=s,selected_groups=selected['groups'],candidates=candidates,train_shots=128*1024,validation_shots=256*1024))
  T=np.random.default_rng(150000+s).uniform(-np.pi,np.pi,(2048,2));ty=target41(T,'annulus')
  for rep in range(10):
   F=noisy_features(T,base,256,np.random.default_rng(151000+s*10000+rep))
   for name,fit in [(f"G{c['groups']}",c['fit']) for c in candidates]+[('validation_selected',selected['fit'])]:rows.append(dict(seed=s,repeat=rep,model=name,**metrics(learned_predict(T,fit,F),ty)))
 log('48_result',{'rows':rows,'fits':fits,'stage45_G4_weight_reconstruction_errors':checks,'decision':'Use equal-shot grouping controls in future tests;compare selected grouping on independent tasks and channels before treating an optimum as universal.'})
if __name__=='__main__':main()
