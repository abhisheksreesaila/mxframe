import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        t = o.get("text", "")
        if isinstance(t, list):
            t = "".join(t)
        if t.strip():
            print(t)
