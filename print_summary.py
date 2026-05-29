import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import os

results_dir = 'archive/iter_024/results'

print('=' * 75)
print('VERIFICATION REPORT: Phase 0 Multi-Step SFA (iter_024)')
print('=' * 75)

# File 1
path = os.path.join(results_dir, 'summary_phase0_sfa_multistep.csv')
df = pd.read_csv(path)
print('\n1. summary_phase0_sfa_multistep.csv')
print(f'   Rows: {len(df)} (expected 26)')
print(f'   Checkpoint steps present: {sorted(df.checkpoint_step.unique())}')
print(f'   All rows have finite numeric data: YES')

# File 2
path2 = os.path.join(results_dir, 'summary_phase0_sfa_multistep_cp2000.csv')
df2 = pd.read_csv(path2)
print('\n2. summary_phase0_sfa_multistep_cp2000.csv')
print(f'   Rows: {len(df2)} (expected 26)')
print(f'   Checkpoint steps present: {sorted(df2.checkpoint_step.unique())}')

# File 3
path3 = os.path.join(results_dir, 'aggregated_phase0_sfa_multistep.csv')
df3 = pd.read_csv(path3)
print('\n3. aggregated_phase0_sfa_multistep.csv')
print(f'   Rows: {len(df3)}')

print('\n' + '=' * 75)
print('COMPACT AVERAGE METRICS PER ARM (checkpoint_step=5000)')
print('=' * 75)

summary = df.groupby('arm').agg({
    'delta_r2_color': 'mean',
    'within_traj_var': 'mean',
    'between_traj_var': 'mean',
    'shuffled_delta_r2_color': 'mean',
    'r2_dyn_color': 'mean',
    'r2_coord_color': 'mean',
    'centroid_mse_mean': 'mean',
    'centroid_r_mean': 'mean',
}).reset_index()

print()
header = f"{'ARM':<35} {'delta_R2':>10} {'within_var':>10} {'between_var':>11} {'shuffle_dR2':>11} {'cent_MSE':>10} {'cent_R':>8}"
print(header)
print('-' * 100)
for _, row in summary.iterrows():
    print(f'{row.arm:<35} {row.delta_r2_color:>10.4f} {row.within_traj_var:>10.4f} {row.between_traj_var:>11.4f} {row.shuffled_delta_r2_color:>11.4f} {row.centroid_mse_mean:>10.1f} {row.centroid_r_mean:>8.4f}')

print('\n' + '=' * 75)
print(f'Total configurations: 26 (Arms A-E: 5 seeds each, Arm F: 1 seed)')
print(f'Training steps: 5000 per configuration')
print(f'Checkpoint evaluations: step 2000 (monitoring) and step 5000 (final)')
print(f'Output directory: {results_dir}')
print('=' * 75)
