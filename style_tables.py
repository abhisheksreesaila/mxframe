import re

with open('BENCHMARKS.md', 'r') as f:
    text = f.read()

# Replace any **number** in the tables with a highlighted yellow box.
# We retain the bold nature via CSS or directly inside the markup.
text = re.sub(r'\*\*((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\*\*', r'<mark style=\
