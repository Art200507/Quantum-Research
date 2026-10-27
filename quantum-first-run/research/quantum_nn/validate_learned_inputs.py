"""Independent checks for learned angles and analytic optimization derivatives."""
from learned_inputs import *

def validate():
 gradient_error=derivative_check();r=np.random.default_rng(128000);X=r.uniform(-4,4,(8,2));W=r.normal(size=(2,2));freq=[.5,1.,2.,3.];F=elementary_features(X@W,freq);errors=[]
 for i,x in enumerate(X):
  for j,f in enumerate(freq):
   S=Statevector.from_instruction(map_circuit(x@W,f));actual=np.array([S.expectation_value(Pauli(p)).real for p in PAULIS]);errors.append(float(abs(F[i,j*15:(j+1)*15]-actual).max()))
 assert max(errors)<1e-12
 derivative=feature_derivatives(X,freq);d_errors=[]
 for k in range(2):
  delta=np.zeros_like(X);delta[:,k]=1e-5;numeric=(elementary_features(X+delta,freq)-elementary_features(X-delta,freq))/(2e-5);d_errors.append(float(abs(numeric-derivative[:,:,k]).max()))
 assert max(d_errors)<1e-7
 return {'passed':True,'joint_gradient_error':gradient_error,'feature_derivative_errors':d_errors,'learned_angle_Qiskit_error':max(errors),'physical_circuits_checked':len(errors)}

if __name__=='__main__':print(validate())
