"""Experiment19: controlled circuit-frequency diagnostic."""
from capabilities import *

def main():
 review();log('19_plan',{'hypothesis':'The re-uploading circuit may have a structural blind spot, rather than merely insufficient training, on the original cosine-product task.','design':'32 random parameter settings,32x32 full angular grid, inspect Fourier power in odd/odd frequency sector required by cosine-product labels; test single-coordinate pi-shift symmetries. Compare basic QNN.','limits':'Numerical diagnostics of this small architecture, not a general QNN expressivity theorem.'})
 x=np.linspace(-np.pi,np.pi,32,endpoint=False);A,B=np.meshgrid(x,x,indexing='ij');X=np.c_[A.ravel(),B.ravel()];r=np.random.default_rng(66000);rows=[]
 odd=np.arange(32)%2==1
 for i in range(32):
  p=r.normal(size=8)
  for model,z in [('basic',qnn_z(X,p[:2])),('reupload',qfeature(X,p,2))]:
   F=np.fft.fft2(z.reshape(32,32))/1024;energy=float((abs(F)**2).sum());oddodd=float((abs(F[np.ix_(odd,odd)])**2).sum())
   if model=='reupload':
    shifted=X.copy();shifted[:,0]+=np.pi;err0=float(np.max(abs(qfeature(shifted,p,2)-z)))
    shifted=X.copy();shifted[:,1]+=np.pi;err1=float(np.max(abs(qfeature(shifted,p,2)-z)))
   else:err0=err1=None
   rows.append({'sample':i,'model':model,'total_energy':energy,'odd_odd_energy':oddodd,'cos_product_coefficient':float(np.mean(z*np.cos(X[:,0])*np.cos(X[:,1]))),'pi_shift_x0_change':err0,'pi_shift_x1_change':err1})
 log('19_result',{'rows':rows,'decision':'Inspect frequency-sector evidence. If adding layers removes earlier useful features, preserve a measured shallow branch and compare a13-parameter QNN mixture with a13-parameter MLP before assuming deeper means more capable.'})

if __name__=='__main__':main()
