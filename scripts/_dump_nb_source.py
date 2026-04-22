import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if src.strip():
        print(f"--- CELL {i} ({c['cell_type']}) ---")
        print(src)
        print()
