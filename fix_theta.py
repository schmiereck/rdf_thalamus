import os

filepath = 'src/run_phase18_experiments.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

theta_char = chr(0x03b8)
old_count = content.count(theta_char)
content = content.replace(theta_char, 'theta')
new_count = content.count(theta_char)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Replaced {old_count} occurrences of theta char. Remaining: {new_count}')
