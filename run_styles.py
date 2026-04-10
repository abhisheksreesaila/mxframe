import re

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Revert if it was already replaced previously to avoid double-wrapping
    text = re.sub(r'<span[^>]*>\*\*MXFrame CPU\*\*</span>', '**MXFrame CPU**', text)
    text = re.sub(r'<span[^>]*>MXFrame GPU</span>', '**MXFrame GPU**', text)
    text = re.sub(r'<span[^>]*>MXFrame CPU</span>', '**MXFrame CPU**', text)
    
    # Process CPU -> yellow text with blue shadow
    # "highlight MX, Graph, CPU with a with a nice yellow, yellow with a blue color"
    text = text.replace('**MXFrame CPU**', '<span style="color:#FFD54F; font-weight:bold; text-shadow: 1px 1px 3px blue;">MXFrame CPU</span>')
    text = text.replace('**mxframe cpu**', '<span style="color:#FFD54F; font-weight:bold; text-shadow: 1px 1px 3px blue;">MXFrame CPU</span>')
    
    # Process GPU -> green coloring + fiery red/orange shadow
    # "MX, GPU with a green color... bright one which choose a color which kind of shows mojos fire"
    text = text.replace('**MXFrame GPU**', '<span style="color:#00E676; font-weight:bold; text-shadow: 1px 1px 5px #FF3D00;">MXFrame GPU</span>')
    text = text.replace('**mxframe gpu**', '<span style="color:#00E676; font-weight:bold; text-shadow: 1px 1px 5px #FF3D00;">MXFrame GPU</span>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

process_file('README.md')
process_file('BENCHMARKS.md')
print("Successfully processed styling!")