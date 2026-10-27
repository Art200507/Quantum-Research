"""Replay frozen experiments to a NEW output directory; review history is JOURNAL.md."""
from pathlib import Path
import argparse,os,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out).resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory.')
out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).parent
settings={**os.environ,'MEMORY_LAB_RESULTS':str(out),'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
for i in range(1,11):
 cmd=[sys.executable,str(root/'lab.py'),str(i)] if i<=7 else [sys.executable,str(root/{8:'extended.py',9:'quantum_check.py',10:'validate.py'}[i])]
 print(f'Replaying {i}/10',flush=True)
 with (out/f'{i:02d}.stdout.txt').open('w') as f:subprocess.run(cmd,env=settings,stdout=f,check=True)
print('Complete:',out)
