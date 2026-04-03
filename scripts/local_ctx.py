#!/usr/bin/env python
"""
Generate LLM context file from documentation.
Combines all doc files into a single llms-ctx.txt for AI assistants.
"""
import re
from pathlib import Path

def generate_local_ctx(llms_file='llms.txt', output_file='llms-ctx.txt'):
    """Generate context file from llms.txt links or fallback to _docs."""
    
    llms_path = Path(llms_file)
    ctx_output = []
    
    # If llms.txt exists, parse links from it
    if llms_path.exists():
        content = llms_path.read_text()
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        
        for link in links:
            local_path = Path(link)
            
            # Convert _proc/ paths to _docs/ paths
            if local_path.parts and local_path.parts[0] == '_proc':
                filename = local_path.name
                # Remove leading digits and underscore (e.g., "00_", "01_")
                if len(filename) > 3 and filename[:2].isdigit() and filename[2] == '_':
                    filename = filename[3:]
                elif len(filename) > 2 and filename[:1].isdigit() and filename[1] == '_':
                    filename = filename[2:]
                local_path = Path('_docs') / filename
            
            if local_path.exists():
                print(f"Processing: {local_path}")
                file_text = local_path.read_text(encoding='utf-8')
                ctx_output.append(f"FILE: {link}\n")
                ctx_output.append(file_text)
                ctx_output.append("\n" + "=" * 40 + "\n")
            else:
                print(f"Skipping: {local_path} (not found)")
    else:
        # Fallback: collect all markdown files from _docs
        docs_path = Path('_docs')
        if docs_path.exists():
            for md_file in sorted(docs_path.glob('*.html.md')):
                print(f"Processing: {md_file}")
                file_text = md_file.read_text(encoding='utf-8')
                ctx_output.append(f"FILE: {md_file}\n")
                ctx_output.append(file_text)
                ctx_output.append("\n" + "=" * 40 + "\n")
    
    if ctx_output:
        # Add header
        header = """# mxframe LLM Context
# GPU-accelerated DataFrames powered by MAX Engine
# Auto-generated - do not edit manually

"""
        Path(output_file).write_text(header + "\n".join(ctx_output), encoding='utf-8')
        print(f"\n✓ Created {output_file}")
    else:
        print("No files found to process")

if __name__ == "__main__":
    generate_local_ctx()
