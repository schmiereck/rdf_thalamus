with open('archive/iter_032/results/analysis.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the two-character string '\\n' with actual newline character '\n'
content = content.replace('\\n', '\n')

with open('archive/iter_032/results/analysis.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Success! Number of lines now:", len(content.split('\n')))
