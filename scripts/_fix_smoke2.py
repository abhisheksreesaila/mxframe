"""Fix the smoke test file in-place using Python."""
path = '/home/ablearn/mxdf_v2/scripts/_test_aot_smoke.py'
with open(path) as f:
    content = f.read()

old = 'df = LazyFrame(Scan(tbl), compiler=comp)\n\n# Grouped sum\nres = df.groupby("key").agg(Expr("col", "val").sum().alias("sum_val")).compute()'
new = 'lf  = LazyFrame(Scan(tbl))\n\n# Grouped sum via compile_and_run (AOT cpu_aot path)\nplan = lf.groupby("key").agg(Expr("col", "val").sum().alias("sum_val")).plan\nres  = comp.compile_and_run(plan)'

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Pattern not found, trying partial...")
    print(repr(content[content.find('df = LazyFrame'):content.find('df = LazyFrame')+200]))
