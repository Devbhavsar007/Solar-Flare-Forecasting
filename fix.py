import re

with open('src/orchestration/agents.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'(\s*except\s+Exception\s+as\s+exc:\s*\n(?:\s*.*\n)*?)(\s*updates\["timing"\]\s*=\s*.*?)\n(\s*return\s+updates)', re.MULTILINE)

def repl(m):
    except_block = m.group(1)
    timing_stmt = m.group(2)
    return_stmt = m.group(3)
    # The indentation of the finally block should match the except block
    indent = except_block.split('except')[0].strip('\n')
    return except_block + indent + 'finally:\n' + indent + '    ' + timing_stmt.lstrip() + '\n' + return_stmt

new_content = pattern.sub(repl, content)

with open('src/orchestration/agents.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
