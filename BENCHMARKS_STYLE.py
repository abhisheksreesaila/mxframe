import re

with open('BENCHMARKS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace any **number** in the tables with a highlighted yellow box.
text = re.sub(r'\*\*((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\*\*', r'<mark style="background-color: #FFEB3B; color: black; font-weight: bold; border-radius: 4px; padding: 0 4px;">\1</mark>', text)

# The user explicitly asked for MX GPU to be a "green color" which kind of shows "mojos fire".
text = text.replace('**MXFrame GPU**', '<span style="color: #00E676; font-weight: bold; text-shadow: 0 0 5px #FF6D00, 1px 1px 3px #FF6D00;">MXFrame GPU</span>')

# Color-code the Winner column entries by framework brand color.
# MXFrame CPU → blue, MXFrame GPU → green, Polars → Polars orange brand color.
text = text.replace('`MXFrame CPU`', '<span style="color:#2196F3; font-weight:bold;">MXFrame CPU</span>')
text = text.replace('`MXFrame GPU`', '<span style="color:#00E676; font-weight:bold;">MXFrame GPU</span>')
text = text.replace('`Polars`', '<span style="color:#FF6600; font-weight:bold;">Polars</span>')

with open('BENCHMARKS.md', 'w', encoding='utf-8') as f:
    f.write(text)
