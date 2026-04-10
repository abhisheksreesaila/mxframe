import re

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Revert if it was already replaced
    text = re.sub(r'<span[^>]*>\*\*MXFrame CPU\*\*</span>', '**MXFrame CPU**', text)
    text = re.sub(r'<span[^>]*>MXFrame GPU</span>', '**MXFrame GPU**', text)
    
    # Process CPU -> yellow text with blue shadow
    text = text.replace('**MXFrame CPU**', '<span style=\
