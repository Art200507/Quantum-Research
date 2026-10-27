"""Replay the logged exploratory sequence into a new log directory."""
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--logs',required=True);a=p.parse_args()
out=Path(a.logs).absolute()
if out.exists() and any(out.iterdir()):raise SystemExit('Use a new/empty log directory.')
out.mkdir(parents=True,exist_ok=True);root=Path(__file__).parent
settings={**os.environ,'QNN_LOG_DIR':str(out),'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
commands=[['experiments.py'],['attention.py','2'],['attention.py','3'],['attention.py','4'],['shots.py'],['noise.py'],['noisy_shots.py'],['value_attention.py'],['robustness.py'],['encoding_range.py'],['hybrid_block.py'],['block_readout.py'],['validate_block.py'],['capabilities.py','16'],['capabilities.py','17'],['matched_control.py'],['expressivity_check.py'],['qnn_mixture.py'],['reliability.py'],['compositional.py'],['generalization_control.py'],['feature_readout.py','25'],['feature_readout.py','26'],['measured_features.py','27'],['measured_features.py','28'],['select_features.py'],['prediction_sets.py'],['corruption_calibration.py'],['feature_control.py'],['new_tasks.py'],['learned_inputs.py'],['encoding_confirmation.py'],['encoder_optimization.py'],['learned_measurements.py'],['encoding_extrapolation.py'],['nonwave_confirmation.py'],['noise_aware_readout.py'],['noise_confirmation.py'],['independent_noise_training.py'],['confidence_control.py'],['device_noise_transfer.py'],['equal_shot_grouping.py'],['grouping_breadth.py'],['affine_control.py'],['asymmetric_drift.py'],['reference_monitor.py'],['triggered_recalibration.py'],['randomized_monitoring.py'],['confidence_guard.py'],['regularized_calibration.py'],['calibration_budget.py'],['fresh_encoders.py'],['temporal_budget.py'],['monitor_blindspot.py']]
for i,args in enumerate(commands,1):
 print('Replaying quantum experiment',i,flush=True)
 with (out/f'{i:02d}_stdout.txt').open('w') as f:subprocess.run([sys.executable,str(root/args[0]),*args[1:]],env=settings,stdout=f,check=True)
