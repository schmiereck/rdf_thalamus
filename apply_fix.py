with open('src/run_phase18_experiments.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'model.load_state_dict(torch.load(cache_path, map_location=device))' in line:
        indent = '        '
        lines.insert(i+1, indent + 'model.d_t = 3  # <-- ADD THIS CRITICAL LINE!\n')
        break

with open('src/run_phase18_experiments.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fix applied!')
