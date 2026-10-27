"""52: channel-matched readout selection versus stale calibration under drift."""
from affine_control import *
from device_noise_transfer import validate as validate_depolarization
ENV52={'A':dict(p=.02,q0=(.01,.08),q1=(.03,.12)),'B':dict(p=.06,q0=(.08,.01),q1=(.12,.03))}
def asymmetric_transition(q0,q1):
 def bit(t):a,b=t;return np.array([[1-a,a],[b,1-b]])
 return np.kron(bit(q1),bit(q0))
def probs52(X,base,env):
 e=ENV52[env];M=asymmetric_transition(e['q0'],e['q1']);P=basis_probabilities(X@np.array(base['input_matrix']),base['frequencies'],BASES)
 return [((1-e['p'])*p+e['p']/4)@M for p in P]
def sample52(P,base,r):
 F,used,covered=measure(P,base['frequencies'],BASES,256,r);assert used==256 and np.all(covered[np.array(base['mask'],bool)]);return F
def validate52():
 err=validate_depolarization()
 for e in ENV52.values():
  M=asymmetric_transition(e['q0'],e['q1']);assert np.all(M>=0) and np.max(abs(M.sum(1)-1))<1e-12
  for i in range(4):
   for j in range(4):
    v=1.
    for bit,t in enumerate((e['q0'],e['q1'])):
     old=(i>>bit)&1;new=(j>>bit)&1;flip=t[old];v*=flip if new!=old else 1-flip
    assert abs(M[i,j]-v)<1e-12
 assert np.array_equal(asymmetric_transition((0,0),(0,0)),np.eye(4))
 return err

def main():
 review();log('52_plan',{'reviewed':'51_result:affine baseline did not erase ring grouping gains;all12 affine fits converge. Other tasks often favor simple calibration.','hypothesis':'Calibration/refitting matched to a device-error distribution improves reliability,but becomes stale when asymmetric readout errors drift.','environments':ENV52,'definition':'Global depolarization once afterCX,then independent asymmetric classical readout flips:q0/q1 tuples=(P(report1|true0),P(report0|true1)). A/B reverse dominant bias and change depolarization. Ideal basis rotations.','design':'Frozen12 encoders from50;for eachdevice A/B measure original128train/256val points4x256shots. Refit G1/G2/G4 andtemperature/affine;select entire pipeline by measured validation CE. Fresh1024test points,5 paired256shot repeats in A and B. Compare50 shot-only selected pipeline,51 shot-only affine,and A/B matched pipelines.','cost':'Per adaptation environment/model:131072training+262144validation shots. Temperature/affine candidates need only validation measurements. Repeats do not add unique labels.','scope':'Hypothetical channels,not IBM calibration or time-series drift. B adaptation assumes fresh representative calibration data available;not automatic drift detection. Channel transfer only,not retraining quantum encoder.'})
 error=validate52();old=json.loads((LOG/'50_result.json').read_text());affines=json.loads((LOG/'51_result.json').read_text());rows=[];fits=[]
 for item in old['fits']:
  s,task,base=item['seed'],item['task'],item['base'];X=np.random.default_rng(131000+s).uniform(-np.pi,np.pi,(1408,2));y=target41(X,task);adapt={};candidates={}
  for i,env in enumerate(ENV52):
   r=np.random.default_rng(166000+s*10000+100*TASKS41.index(task)+i*1000);Ptr=probs52(X[:128],base,env);Pva=probs52(X[128:384],base,env);blocks=np.stack([sample52(Ptr,base,r) for _ in range(4)]);V=np.concatenate([sample52(Pva,base,r) for _ in range(4)]);vy=np.tile(y[128:384],4);cs=candidates_from_blocks(blocks,y[:128],V,vy,base);vz=learned_predict(np.zeros((len(V),2)),base,V);a=affine_fit(vz,vy);cs.append(dict(model='affine',fit=base,inverse_temperature=a['scale'],bias=a['bias'],selection_loss=a['validation_loss'],affine_status=a));candidates[env]=cs;adapt[env]=min(cs,key=lambda c:c['selection_loss'])
  stale=next(c for c in item['candidates'] if c['model']==item['selected_model']);a=next(f for f in affines['fits'] if f['seed']==s and f['task']==task);clean_affine=dict(fit=base,inverse_temperature=a['scale'],bias=a['bias']);fits.append(dict(seed=s,task=task,selected_A=adapt['A']['model'],selected_B=adapt['B']['model'],candidates=candidates))
  T=np.random.default_rng(167000+s).uniform(-np.pi,np.pi,(1024,2));ty=target41(T,task)
  for i,env in enumerate(ENV52):
   P=probs52(T,base,env)
   for rep in range(5):
    F=sample52(P,base,np.random.default_rng(168000+s*10000+100*TASKS41.index(task)+i*1000+rep))
    for name,c in [('shot_only_selected',stale),('shot_only_affine',clean_affine),('adapted_A',adapt['A']),('adapted_B',adapt['B'])]:rows.append(dict(seed=s,task=task,environment=env,repeat=rep,model=name,**metrics(learned_predict(T,c['fit'],F)*c['inverse_temperature']+c.get('bias',0.),ty)))
  print(s,task,flush=True)
 log('52_result',{'rows':rows,'fits':fits,'depolarization_Qiskit_max_error':error,'transition_validation':'Row stochastic,identity limit andall entries checked against independent bit enumeration.','decision':'Compare matched versus stale calibration per task;new device monitoring question only if representative recalibration helps consistently. No claim of predicting or detecting drift.'})
if __name__=='__main__':main()
