"""46: positive scalar logit calibration versus noisy-feature refitting."""
from independent_noise_training import *
from scipy.optimize import minimize_scalar

def main():
 review();old=json.loads((LOG/'45_result.json').read_text())
 log('46_plan',{'reviewed':'45_result: all5 independent seeds improve low-shot CE;mean accuracy84.4->87.6%,confidently-wrong8.4->2.3%. High-shot mixed.','hypothesis':'A cheaper confidence-only adjustment might explain most reliability gains.','design':'Freeze45 models. Fit one positive inverse-temperature scalar on exactly the saved noisy validation observations used by noisy refit. Compare ideal,temperature and noise readouts on new2048test points per seed,10 identical paired measurement draws,256/1024shots. No retraining or test tuning.','cost':'Temperature uses same1024*B validation shots,zero noisy training shots versus512*B for noisy readout. Shares already ideal-trained encoder. Validation labels were also used in encoder selection,so this is model selection,not an independent calibration guarantee.','metrics':'CE,Brier,accuracy,confidently wrong;temperature cannot change positive/negative classification.'})
 rows=[];fits=[]
 for item in old['fits']:
  s,B,base=item['seed'],item['budget'],item['base'];V=np.array(item['validation_features']);vy=np.array(item['validation_labels']);vz=learned_predict(np.zeros((len(V),2)),base,V)
  def loss(a):return float(np.mean(np.logaddexp(0,a*vz)-vy*a*vz))
  result=minimize_scalar(loss,bounds=(.02,20),method='bounded');a=float(result.x);assert result.success
  fits.append(dict(seed=s,budget=B,inverse_temperature=a,validation_loss=result.fun,success=bool(result.success)))
  T=np.random.default_rng(145000+s).uniform(-np.pi,np.pi,(2048,2));y=target41(T,'annulus')
  for rep in range(10):
   F=noisy_features(T,base,B,np.random.default_rng(146000+s*10000+B+rep));z=learned_predict(T,base,F);zn=learned_predict(T,item['noise_fit'],F);assert np.array_equal(z>0,a*z>0)
   for name,zz in [('ideal',z),('temperature',a*z),('noise',zn)]:rows.append(dict(seed=s,budget=B,repeat=rep,model=name,**metrics(zz,y)))
 log('46_result',{'rows':rows,'fits':fits,'decision':'Separate reduced overconfidence from improved decision boundaries;compare cheap scalar control before proposing a hardware research question.'})
if __name__=='__main__':main()
