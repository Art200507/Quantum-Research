"""Experiment23: challenge held-out-region findings with a generic Fourier model."""
from compositional import *

def main():
 review();prior=json.loads((LOG/'22_result.json').read_text())
 log('23_plan',{'hypothesis':'A QNN beating raw-input MLPs on a withheld region may reflect a useful periodic-feature prior rather than quantum-specific generalization. Test generic Fourier features on identical splits.','design':'Same3 seeds/3 tasks from22; reuse the already defined generic frequency0..3 basis and regularizer from16, no new hyperparameter choices or held-out-region tuning.','limits':'51-parameter Fourier control versus9-parameter QNN mixture is a stronger unequal-size baseline; report counts, not capacity matching. Its inclusion is motivated by observed outcomes and remains exploratory.','reviewed_previous_rows':len(prior['rows'])})
 rows=[];fits=[]
 for seed in range(3):
  pool=np.random.default_rng(71000+seed).uniform(-np.pi,np.pi,(4096,2));seen=pool[~((pool[:,0]>0)&(pool[:,1]>0))]
  pool=np.random.default_rng(72000+seed).uniform(-np.pi,np.pi,(4096,2));held=pool[(pool[:,0]>0)&(pool[:,1]>0)][:512]
  trainX=seen[:128];valX=seen[128:256];testX=seen[256:768]
  for task in ('cos_product','radial','mixed_frequency'):
   train=(trainX,labels(trainX,task));val=(valX,labels(valX,task));fit,predict=fit_model('fourier',train,val,seed)
   fits.append({'seed':seed,'task':task,'model':'fourier',**fit})
   for split,X in [('train',trainX),('validation',valX),('seen_test',testX),('held_out_quadrant',held)]:rows.append({'seed':seed,'task':task,'model':'fourier','split':split,**evaluate(predict(X,fit['parameters']),labels(X,task))})
 log('23_result',{'rows':rows,'fits':fits,'decision':'Keep every task and control in the generalization comparison. Quantum circuit bias may help selected tasks; do not equate a narrow win over raw-input MLPs with broader intelligence or a unique quantum capability.'})

if __name__=='__main__':main()
