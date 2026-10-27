"""Compile locally and verify known ideal outputs, without account access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeTorino
import mirror
out=Path(__file__).resolve().parents[1]/'results'
backend=FakeTorino(); rows=[]
for n in (5,20,50,100):
 for barrier in (False,True):
  qc=mirror._build_mirror_circuit(n,8,1234+n)
  if not barrier:
   qc.data=[i for i in qc.data if i.operation.name!='barrier']
  level=3 if n<=20 else 1
  isa=transpile(qc,backend,optimization_level=level,seed_transpiler=17)
  ops=dict(isa.count_ops())
  rows.append({'n':n,'barrier':barrier,'optimization_level':level,'seed_transpiler':17,'depth':isa.depth(),'ops':ops,'two_qubit_gates':sum(i.operation.num_qubits==2 for i in isa.data if i.operation.name not in ('barrier','measure'))})
# Isolate cancellation from routing using a fixed all-to-all basis compiler.
for barrier in (False,True):
 qc=mirror._build_mirror_circuit(5,8,1239).remove_final_measurements(inplace=False)
 if not barrier:qc.data=[i for i in qc.data if i.operation.name!='barrier']
 isa=transpile(qc,basis_gates=['u','cx'],optimization_level=3,seed_transpiler=17)
 prob=float(abs(Statevector.from_instruction(isa).data[0])**2)
 assert abs(prob-1)<1e-10
 rows.append({'n':5,'barrier':barrier,'target':'all-to-all u/cx','depth':isa.depth(),'ops':dict(isa.count_ops()),'ideal_return_probability':prob})
(out/'compiled_mirrors.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
