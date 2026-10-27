"""Make a standalone figure and record experiment/source provenance."""
from pathlib import Path
import hashlib,importlib.metadata,json,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent;out=ROOT/'results'
p=json.loads((out/'10.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,sigma in zip(axs,(.3,.8)):
 for method,label in [('bind','Phase binding'),('linear8','Linear, int8'),('ridge8','Trained linear, int8'),('exact8','Nearest key, int8')]:
  r=[x for x in p['summary'] if x['sigma']==sigma and x['method']==method]
  ax.errorbar([x['n'] for x in r],[100*x['accuracy_mean'] for x in r],yerr=[100*1.96*x['seed_se'] for x in r],marker='o',capsize=3,label=label)
 ax.set(xlabel='Stored associations',ylabel='Recall accuracy (%)',ylim=(0,105),title=f'Query phase noise sigma={sigma} radians')
 ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('24 fresh random memories • 16 classes • classical computation')
fig.tight_layout();fig.savefig(out/'heldout.png',dpi=180);plt.close(fig)
files=list(ROOT.glob('*.py'))+list(out.glob('[0-9][0-9].json'))
meta={'date':'2026-10-26','python':platform.python_version(),'platform':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','qiskit','qiskit-aer','matplotlib']},'rows_per_experiment':{i:len(json.loads((out/f'{i:02d}.json').read_text())['rows']) for i in range(1,11)},'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'no_hardware':True,'no_energy_measurement':True,'figure_error_bars':'1.96 * between-memory standard error; descriptive normal approximation, not simultaneous intervals; zero empirical SE does not guarantee future perfection'}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta['rows_per_experiment']))
