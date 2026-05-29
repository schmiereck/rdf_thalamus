import pandas as pd
import os
import numpy as np

results_dir = 'archive/iter_024/results'
files = {
    'summary_phase0_sfa_multistep.csv': os.path.join(results_dir, 'summary_phase0_sfa_multistep.csv'),
    'summary_phase0_sfa_multistep_cp2000.csv': os.path.join(results_dir, 'summary_phase0_sfa_multistep_cp2000.csv'),
    'aggregated_phase0_sfa_multistep.csv': os.path.join(results_dir, 'aggregated_phase0_sfa_multistep.csv'),
}

print('=' * 70)
print('VERIFICATION OF OUTPUT FILES')
print('=' * 70)

for name, path in files.items():
    print(f'\n--- {name} ---')
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'  Exists: YES')
        print(f'  Rows: {len(df)}')
        print(f'  Columns: {len(df.columns)}')
        if 'checkpoint_step' in df.columns:
            steps = df['checkpoint_step'].unique()
            print(f'  Checkpoint steps: {steps}')
    else:
        print(f'  Exists: NO')

print('\n' + '=' * 70)
print('FINAL SUMMARY (checkpoint_step=5000)')
print('=' * 70)

df_final = pd.read_csv(files['summary_phase0_sfa_multistep.csv'])
# Filter to only step 5000 final results
df_final = df_final[df_final['checkpoint_step'] == 5000]
print(f'Final results rows: {len(df_final)}')

# Group by arm and compute means
numeric_cols = [
    'delta_r2_color', 'within_traj_var', 'between_traj_var', 
    'shuffled_delta_r2_color', 'r2_dyn_color', 'r2_coord_color',
    'centroid_mse_mean', 'centroid_r_mean', 'mean_dyn_delta', 'mean_coord_delta',
    'slowness_ratio', 'tracking_delta_corr', 'tracking_level_corr',
    'normalized_dyn_var', 'normalized_coord_var',
    'shuffled_r2_dyn_color', 'shuffled_r2_dyn_identity', 'shuffled_delta_r2_identity'
]

present_cols = [c for c in numeric_cols if c in df_final.columns]
agg = df_final.groupby('arm')[present_cols].mean().reset_index()

print('\n' + '-' * 70)
for _, row in agg.iterrows():
    print(f'\nARM: {row["arm"]}')
    for col in present_cols:
        val = row[col]
        if pd.notna(val):
            print(f'  {col:30s}: {val:10.6f}')
        else:
            print(f'  {col:30s}: N/A')

print('\n' + '=' * 70)
print('CHECKPOINT 2000 SUMMARY')
print('=' * 70)

df_cp = pd.read_csv(files['summary_phase0_sfa_multistep_cp2000.csv'])
print(f'Checkpoint 2000 rows: {len(df_cp)}')
agg_cp = df_cp.groupby('arm')[present_cols].mean().reset_index()

print('\n' + '-' * 70)
for _, row in agg_cp.iterrows():
    print(f'\nARM: {row["arm"]}')
    for col in present_cols:
        val = row[col]
        if pd.notna(val):
            print(f'  {col:30s}: {val:10.6f}')
        else:
            print(f'  {col:30s}: N/A')

print('\nDone!')
