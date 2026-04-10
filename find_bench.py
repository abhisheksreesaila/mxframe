import json
import glob

def search_notebooks():
    for f in glob.glob('benchmarks/*.ipynb'):
        with open(f, 'r', encoding='utf-8') as nb:
            data = json.load(nb)
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    for out in cell.get('outputs', []):
                        if 'text' in out:
                            text = ''.join(out['text'])
                            if 'ms' in text or 'seconds' in text:
                                print(f"--- Found output in {f} ---")
                                print(text)


search_notebooks()
