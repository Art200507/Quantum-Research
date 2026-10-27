"""59: held-out selection of affine shrinkage versus a fixed confidence cap."""
from confidence_guard import *
LAMBDAS=(0.,.0001,.001,.01,.1)

def shrink_fit(z,y,lam):
 def objective(p):
  t=p[0]*z+p[1];d=(expit(t)-y)/len(y);prior=p-np.array([1.,0.])
  return float(np.mean(np.logaddexp(0,t)-y*t)+lam*(prior@prior)),np.array([d@z,d.sum()])+2*lam*prior
 res=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.02,20),(-10,10)],options={'maxiter':200,'ftol':1e-10})
 return dict(scale=float(res.x[0]),bias=float(res.x[1]),lambda_=lam,success=bool(res.success),iterations=res.nit)
def cal_predict(z,f):return f['scale']*z+f['bias']
def main():
 review();log('59_plan',{'reviewed':'58_checkpoint and57_result:extremeconfidence aftersmallnoisycalibration;fixedcaphelpsCEbuthurtsBrierslightly. Testregularizedcalibration withindependentselectionexamples.','hypothesis':'Held-outselectedshrinkage towardidentityaffinecorrection reducesextremeprobabilityerrors beyondMLEcalibration,andmaycompete withfixedcap.','design':'12frozenencoders,channelsA/B,5freshindependentcalibration/testcohorts each.128freshknowncalibrationlabels:64fit/64selectlambda onCE;thenrefitselectedlambda onall128. CompareMLEall128,MLEall128cap4,regularizedall128,regularizedall128cap4. Identical256shotfeatures;2048fresh testpoints/cohort. Calibration/trainencoder/testlabelsdisjoint.','lambda_grid':LAMBDAS,'objective':'mean logisticCE+lambda*((scale-1)^2+bias^2),sameaffineboundsas51. Selectlambdausingheldout64labels,notasymptoticcoverage.','cost':'Allmethods use32768calibrationshots/model/cohort and524288inferenceshots;classicalfit/search costsnotcounted asquantumshots. RawMLEcanuseall128directly. Refitafterselectionstandardizes totalfitlabels.','scope':'Staticchannelcalibration only;oldencoders reused,synthetic2Dtasks;notyetend-to-endmonitoringimprovement ornovelquantumalgorithm.'})
 rows=[];fits=[]
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task)
  for env in ('A','B'):
   for rep in range(5):
    r=np.random.default_rng(211000+s*100000+tid*10000+(env=='B')*1000+rep);C=r.uniform(-np.pi,np.pi,(128,2));cy=target41(C,task);F=sample52(probs52(C,base,env),base,r);z=learned_predict(C,base,F);cs=[]
    for lam in LAMBDAS:
     f=shrink_fit(z[:64],cy[:64],lam);v=cal_predict(z[64:],f);f['selection_loss']=float(np.mean(np.logaddexp(0,v)-cy[64:]*v));cs.append(f)
    selected=min(cs,key=lambda f:f['selection_loss']);raw=shrink_fit(z,cy,0.);regular=shrink_fit(z,cy,selected['lambda_']);fits.append(dict(seed=s,task=task,environment=env,repeat=rep,candidates=cs,raw=raw,regularized=regular))
    T=r.uniform(-np.pi,np.pi,(2048,2));y=target41(T,task);zt=learned_predict(T,base,sample52(probs52(T,base,env),base,r))
    for name,fit,cap in [('MLE',raw,False),('MLE_cap4',raw,True),('regularized',regular,False),('regularized_cap4',regular,True)]:
     zz=cal_predict(zt,fit);zz=np.clip(zz,-4,4) if cap else zz;rows.append(dict(seed=s,task=task,environment=env,repeat=rep,model=name,lambda_=fit['lambda_'],calibration_shots=32768,inference_shots=524288,total_shots=557056,**metrics(zz,y)))
  print(s,task,flush=True)
 log('59_result',{'rows':rows,'fits':fits,'decision':'Reviewregularization andcap bytask/channel,includingregressions. Choosenextconsolidatedbenchmark fromtheseobservations;do notassume staticcalibration gains transfer totemporalmonitoring.'})
if __name__=='__main__':main()
