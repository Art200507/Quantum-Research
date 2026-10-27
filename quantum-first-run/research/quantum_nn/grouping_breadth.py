"""50: fixed-shot grouping and calibrated baselines across non-wave tasks."""
from confidence_control import *

def candidates_from_blocks(blocks,y,V,vy,base):
 candidates=[dict(model='ideal',fit=base,inverse_temperature=1.)]
 vz=learned_predict(np.zeros((len(V),2)),base,V)
 res=minimize_scalar(lambda a:float(np.mean(np.logaddexp(0,a*vz)-vy*a*vz)),bounds=(.02,20),method='bounded');assert res.success
 candidates.append(dict(model='temperature',fit=base,inverse_temperature=float(res.x)))
 for G in (1,2,4):
  F=blocks.reshape(G,4//G,len(y),-1).mean(1).reshape(G*len(y),-1)
  candidates.append(dict(model=f'G{G}',fit=refit(F,np.tile(y,G),V,vy,base),inverse_temperature=1.))
 for c in candidates:
  z=learned_predict(np.zeros((len(V),2)),c['fit'],V)*c['inverse_temperature'];c['selection_loss']=float(np.mean(np.logaddexp(0,z)-vy*z))
 return candidates

def main():
 review();log('50_plan',{'reviewed':'49_checkpoint and48_result: grouping benefits established mainly on ring;temperature competitive and training cost must remain visible.','hypothesis':'Grouping benefits may extend to other boundaries,but a validation-selected pipeline may need to retain temperature/ideal options.','design':'All12 frozen stage41 learned encoders(3 original training seeds,4 tasks). Reuse original128train/256validation labels;new measurement records,4x256shots per point. G1/G2/G4 reuse identical1024-shot training record;validation4 independent256shot observations/point. Positive temperature and ideal controls. Choose entire pipeline by noisy validation CE. Fresh1024test points,5 paired256shot measurement repeats.','cost':'Grouping methods each131072training shots+262144validation shots/model;temperature and ideal need zero extra noisy training shots. Encoder fitting ideal and shared.','scope':'Fresh measurement/test samples,not new encoders;per-task models,2qubits,no gate noise,no uniquely quantum capability. No extra labels from repeats.'})
 old=json.loads((LOG/'41_result.json').read_text());rows=[];fits=[]
 for item in old['fits']:
  s,task,base=item['seed'],item['task'],item['learned'];X=np.random.default_rng(131000+s).uniform(-np.pi,np.pi,(1408,2));y=target41(X,task);r=np.random.default_rng(161000+s*10000+100*TASKS41.index(task));blocks=np.stack([noisy_features(X[:128],base,256,r) for _ in range(4)]);V=np.concatenate([noisy_features(X[128:384],base,256,r) for _ in range(4)]);vy=np.tile(y[128:384],4);cs=candidates_from_blocks(blocks,y[:128],V,vy,base);selected=min(cs,key=lambda c:c['selection_loss']);fits.append(dict(seed=s,task=task,base=base,candidates=cs,selected_model=selected['model']))
  T=np.random.default_rng(162000+s).uniform(-np.pi,np.pi,(1024,2));ty=target41(T,task)
  for rep in range(5):
   F=noisy_features(T,base,256,np.random.default_rng(163000+s*10000+100*TASKS41.index(task)+rep))
   for name,c in [(c['model'],c) for c in cs]+[('validation_selected',selected)]:rows.append(dict(seed=s,task=task,repeat=rep,model=name,**metrics(learned_predict(T,c['fit'],F)*c['inverse_temperature'],ty)))
  print(s,task,flush=True)
 log('50_result',{'rows':rows,'fits':fits,'decision':'Review every task and selected pipeline before channel-aware follow-up;retain negative results and avoid recommending grouping everywhere.'})
if __name__=='__main__':main()
