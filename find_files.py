import os

def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception as e:
        return f"Error: {e}"

print("=== ALL FILES IN archive/iter_004/ ===")
for root, dirs, files in os.walk('archive/iter_004'):
    for file in files:
        full_path = os.path.join(root, file)
        normalized_path = full_path.replace("\\", "/")
        line_count = count_lines(full_path)
        print(f"{normalized_path} (lines: {line_count})")
