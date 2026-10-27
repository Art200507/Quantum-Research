"""60: paired calibration-label/shot budget sweep with shared test records."""
from regularized_calibration import *
BUDGETS=(64,128,256)
def main():
 review();log('60_plan',{'reviewed':'59_result:128labelMLE meanCE .20352 vsfixedcap .20967;selectedshrinkage .20384,slightlyloweraccuracy. Contradictsuniversalconfidencecapbenefit. Needpairedlabel-budgettest beforeattributingchange tomoredata.','hypothesis':'Calibration sample size changes thereliability/costfrontier;regularization orcappingmayonlyhelpinsmall-dataregimes.','design':'12frozenencoders,channelsA/B,5freshcohorts.256freshcalibrationpoints with256shotfeatures perpoint. Nestedprefix64/128/256labels;eachbudgetsplit half fit/halfvalidation toselectlambda,thenrefitallbudgetlabels. CompareMLE,MLEcap4,selectedregularized,selectedregularizedcap4. Same2048testpointsandmeasurementrecord for allbudgets/methodswithincohort.','cost':'16384/32768/65536calibrationshots plus524288inferenceshots. Refitting/classicalselection notextraquantumshots. Differentbudgetslabeledexplicitly,notclaimed equal-cost.','primary':'PairedCE/Brier/accuracyperbudget;retainallmodels andseeds,no test-driven choiceofcap/lambda.','scope':'Staticchannels,nottemporalmonitoring. 5calibrationcohorts are notnewencodertrainingseeds. Exactclassical2qubitequivalent remains.'})
 rows=[];fits=[]
 for item in json.loads((LOG/'50_result.json').read_text())['fits']:
  s,task,base=item['seed'],item['task'],item['base'];tid=TASKS41.index(task)
  for env in ('A','B'):
   for rep in range(5):
    r=np.random.default_rng(221000+s*100000+tid*10000+(env=='B')*1000+rep);C=r.uniform(-np.pi,np.pi,(256,2));cy=target41(C,task);zc=learned_predict(C,base,sample52(probs52(C,base,env),base,r));T=r.uniform(-np.pi,np.pi,(2048,2));y=target41(T,task);zt=learned_predict(T,base,sample52(probs52(T,base,env),base,r))
    for N in BUDGETS:
     half=N//2;cs=[]
     for lam in LAMBDAS:
      f=shrink_fit(zc[:half],cy[:half],lam);v=cal_predict(zc[half:N],f);f['selection_loss']=float(np.mean(np.logaddexp(0,v)-cy[half:N]*v));cs.append(f)
     chosen=min(cs,key=lambda f:f['selection_loss']);raw=shrink_fit(zc[:N],cy[:N],0.);reg=shrink_fit(zc[:N],cy[:N],chosen['lambda_']);fits.append(dict(seed=s,task=task,environment=env,repeat=rep,calibration_labels=N,candidates=cs,raw=raw,regularized=reg))
     for name,f,cap in [('MLE',raw,False),('MLE_cap4',raw,True),('regularized',reg,False),('regularized_cap4',reg,True)]:
      z=cal_predict(zt,f);z=np.clip(z,-4,4) if cap else z;rows.append(dict(seed=s,task=task,environment=env,repeat=rep,calibration_labels=N,model=name,calibration_shots=N*256,inference_shots=524288,total_shots=N*256+524288,**metrics(z,y)))
  print(s,task,flush=True)
 log('60_result',{'rows':rows,'fits':fits,'decision':'Reviewnestedbudgeteffects separatelyfrommethodchoice. Chooseanexplicitdeploymentbudgetonlyaftertemporalconfirmationandfreshtrainingseeds;do notcarryforwardthecapasdefault.'})
if __name__=='__main__':main()
