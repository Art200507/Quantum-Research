"""51: two-parameter positive-affine calibration as a stronger cheap control."""
from grouping_breadth import *

def affine_fit(z,y):
 def objective(p):
  t=p[0]*z+p[1];d=(expit(t)-y)/len(y)
  return float(np.mean(np.logaddexp(0,t)-y*t)),np.array([d@z,d.sum()])
 res=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.02,20),(-10,10)],options={'maxiter':200,'ftol':1e-10})
 return dict(scale=float(res.x[0]),bias=float(res.x[1]),validation_loss=float(res.fun),success=bool(res.success),iterations=res.nit)
def restore_validation(item):
 s,task,base=item['seed'],item['task'],item['base'];X=np.random.default_rng(131000+s).uniform(-np.pi,np.pi,(1408,2));r=np.random.default_rng(161000+s*10000+100*TASKS41.index(task))
 for _ in range(4):noisy_features(X[:128],base,256,r)
 V=np.concatenate([noisy_features(X[128:384],base,256,r) for _ in range(4)])
 return V,np.tile(target41(X[128:384],task),4)
def main():
 review();log('51_plan',{'reviewed':'50_result: grouping ring gains modest;temperature usually stronger on other3 tasks. Temperature cannot move boundary,so affine logit correction is a missing cheap control.','hypothesis':'A positive scale plus intercept may explain decision improvements previously attributed to full noisy readout refitting.','design':'Freeze50 models andsame noisy validation measurements;fit a*z+b witha>0,compare ideal,temperature,affine,and50 validation-selected pipeline on fresh1024test points,5 paired256shot repeats for12models.','cost':'Affine uses262144 validation shots and no noisy training shots. Group candidates additionally need131072training shots. Two free affine parameters fit on256unique validation labels(4repeats).','scope':'All stages exploratory;validation used for encoder selection and calibration,not conformal coverage;test labels untouched. Affine calibration is an established classical baseline,not a new quantum method.'})
 rows=[];fits=[]
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];V,vy=restore_validation(item);vz=learned_predict(np.zeros((len(V),2)),base,V);a=affine_fit(vz,vy);fits.append(dict(seed=s,task=task,**a));temp=next(c for c in item['candidates'] if c['model']=='temperature');chosen=next(c for c in item['candidates'] if c['model']==item['selected_model']);T=np.random.default_rng(164000+s).uniform(-np.pi,np.pi,(1024,2));y=target41(T,task)
  for rep in range(5):
   F=noisy_features(T,base,256,np.random.default_rng(165000+s*10000+100*TASKS41.index(task)+rep));z=learned_predict(T,base,F)
   for name,zz in [('ideal',z),('temperature',temp['inverse_temperature']*z),('affine',a['scale']*z+a['bias']),('prior_selected',learned_predict(T,chosen['fit'],F)*chosen['inverse_temperature'])]:rows.append(dict(seed=s,task=task,repeat=rep,model=name,**metrics(zz,y)))
 log('51_result',{'rows':rows,'fits':fits,'decision':'Retain affine calibration as a serious low-cost comparator. Next test asymmetric device/readout errors and whether recalibration transfers across drift.'})
if __name__=='__main__':main()
