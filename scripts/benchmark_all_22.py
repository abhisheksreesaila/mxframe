#!/usr/bin/env python3
"""Memory-bounded TPC-H benchmark: MXFrame CPU/GPU, Polars, and Pandas."""
import argparse
import gc
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import polars as pl

from scripts.benchmark_tpch import (
    make_lineitem,
    make_tpch_q3_tables, make_tpch_q12_tables, make_tpch_q14_tables,
    make_tpch_q5_tables, make_tpch_q10_tables,
    make_tpch_q7_tables, make_tpch_q8_tables,
    make_tpch_q13_tables, make_tpch_q19_tables,
    make_tpch_q4_tables, make_tpch_q9_tables,
    make_tpch_q11_tables, make_tpch_q18_tables, make_tpch_q16_tables,
    make_tpch_q17_tables, make_tpch_q2_tables, make_tpch_q15_tables,
    make_tpch_q20_tables, make_tpch_q21_tables, make_tpch_q22_tables,
    run_q1_mxframe,  run_q1_pandas,  run_q1_polars,
    run_q6_mxframe,  run_q6_pandas,  run_q6_polars,
    run_q3_mxframe,  run_q3_pandas,  run_q3_polars,
    run_q12_mxframe, run_q12_pandas, run_q12_polars,
    run_q14_mxframe, run_q14_pandas, run_q14_polars,
    run_q5_mxframe,  run_q5_pandas,  run_q5_polars,
    run_q10_mxframe, run_q10_pandas, run_q10_polars,
    run_q7_mxframe,  run_q7_pandas,  run_q7_polars,
    run_q8_mxframe,  run_q8_pandas,  run_q8_polars,
    run_q13_mxframe, run_q13_pandas, run_q13_polars,
    run_q19_mxframe, run_q19_pandas, run_q19_polars,
    run_q4_mxframe,  run_q4_pandas,  run_q4_polars,
    run_q9_mxframe,  run_q9_pandas,  run_q9_polars,
    run_q11_mxframe, run_q11_pandas, run_q11_polars,
    run_q18_mxframe, run_q18_pandas, run_q18_polars,
    run_q16_mxframe, run_q16_pandas, run_q16_polars,
    run_q17_mxframe, run_q17_pandas, run_q17_polars,
    run_q2_mxframe,  run_q2_pandas,  run_q2_polars,
    run_q15_mxframe, run_q15_pandas, run_q15_polars,
    run_q20_mxframe, run_q20_pandas, run_q20_polars,
    run_q21_mxframe, run_q21_pandas, run_q21_polars,
    run_q22_mxframe, run_q22_pandas, run_q22_polars,
)
from mxframe.custom_ops import clear_cache

ALL_QUERIES = list(range(1, 23))
DESCRIPTIONS = {
    1:"Filter+8Agg", 2:"Min Cost Supplier", 3:"3-Table Join+Agg",
    4:"Order Priority", 5:"Multi-Join+GroupBy", 6:"Masked GlobalAgg",
    7:"Shipping Volume", 8:"Market Share", 9:"Product Profit",
    10:"Customer Revenue", 11:"Important Stock", 12:"2-Table Join+Agg",
    13:"Customer Distrib", 14:"Promo Revenue", 15:"Top Supplier Rev",
    16:"Part/Supplier Rel", 17:"Small Qty Order", 18:"Large Volume Cust",
    19:"Discounted Revenue", 20:"Potential Part Promo",
    21:"Suppliers Who Kept", 22:"Global Sales Oppty",
}


def parse_queries(value):
    queries = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            queries.extend(range(start, end + 1))
        else:
            queries.append(int(part))
    invalid = sorted(set(queries) - set(ALL_QUERIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"queries must be in Q1-Q22: {invalid}")
    return list(dict.fromkeys(queries))

def synchronize_gpu():
    try:
        import cupy
        cupy.cuda.get_current_stream().synchronize()
    except ImportError:
        pass


def t(fn, warmup=1, runs=3, synchronize=False):
    try:
        for _ in range(warmup):
            fn()
            if synchronize: synchronize_gpu()
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            if synchronize: synchronize_gpu()
            samples.append((time.perf_counter()-t0)*1000.0)
        samples.sort()
        return round(samples[len(samples)//2], 1)
    except Exception as e:
        return f"ERR:{str(e)[:40]}"

def bench_scale(N, s=1, selected=None, output=None, rapids_only=False, engine=None):
    selected = set(selected or ALL_QUERIES)
    print(f"\n{'='*75}\n  Benchmarking at {N:,} rows  (scale factor s={s})\n{'='*75}")
    print(f"  Generating data for {', '.join(f'Q{q}' for q in sorted(selected))}...", end="", flush=True)
    data = {}
    if selected & {1, 6}: data["li"] = make_lineitem(N)
    if 2 in selected: data[2] = make_tpch_q2_tables()
    if 3 in selected: data[3] = make_tpch_q3_tables(15000*s,150000*s,600000*s)
    if 4 in selected: data[4] = make_tpch_q4_tables(150000*s,600000*s)
    if 5 in selected: data[5] = make_tpch_q5_tables(15000*s,100000*s,400000*s)
    if 7 in selected: data[7] = make_tpch_q7_tables(n_sup=2000*s,n_cust=10000*s,n_orders=80000*s,n_li=200000*s)
    if 8 in selected: data[8] = make_tpch_q8_tables(n_parts=5000*s,n_cust=10000*s,n_orders=80000*s,n_li=200000*s)
    if 9 in selected: data[9] = make_tpch_q9_tables(5000*s,2000*s,80000*s,200000*s)
    if 10 in selected: data[10] = make_tpch_q10_tables(15000*s,150000*s,600000*s)
    if 11 in selected: data[11] = make_tpch_q11_tables()
    if 12 in selected: data[12] = make_tpch_q12_tables(50000*s,300000*s)
    if 13 in selected: data[13] = make_tpch_q13_tables(150000*s,600000*s)
    if 14 in selected: data[14] = make_tpch_q14_tables(20000*s,200000*s)
    if 15 in selected: data[15] = make_tpch_q15_tables(2000*s,200000*s)
    if 16 in selected: data[16] = make_tpch_q16_tables()
    if 17 in selected: data[17] = make_tpch_q17_tables(5000*s,200000*s)
    if 18 in selected: data[18] = make_tpch_q18_tables(15000*s,60000*s,300000*s)
    if 19 in selected: data[19] = make_tpch_q19_tables(20000*s,200000*s)
    if 20 in selected: data[20] = make_tpch_q20_tables(5000*s,2000*s,100000*s)
    if 21 in selected: data[21] = make_tpch_q21_tables(2000*s,60000*s,200000*s)
    if 22 in selected: data[22] = make_tpch_q22_tables(150000*s,400000*s)
    print(" done")

    rows = []
    def row(qname, desc, cpu_fn, gpu_fn, pl_fn, pd_fn):
        print(f"  {qname:<5}", end="", flush=True)
        if rapids_only:
            result = t(pd_fn, synchronize=True)
            rows.append({"Q": qname, "RAPIDS-cuDF(ms)": result})
            print(f"  rapids-cudf={result}ms")
            gc.collect()
            return

        if engine is not None:
            functions = {
                "mx-cpu": (cpu_fn, "MX-CPU(ms)"),
                "mx-gpu": (gpu_fn, "MX-GPU(ms)"),
                "polars": (pl_fn, "Polars(ms)"),
                "pandas": (pd_fn, "Pandas(ms)"),
            }
            engine_fn, column = functions[engine]
            result = "N/A:JIT" if engine == "mx-gpu" and qname == "Q8" else t(engine_fn)
            rows.append({"Q": qname, "Description": desc, column: result})
            print(f"  {engine}={result}ms")
            clear_cache()
            gc.collect()
            return

        gpu_result = "N/A:JIT"
        if qname != "Q8":
            gpu_result = t(gpu_fn)
        r = {"Q":qname,"Description":desc,
                 "MX-CPU(ms)":t(cpu_fn),"MX-GPU(ms)":gpu_result,
             "Polars(ms)":t(pl_fn),"Pandas(ms)":t(pd_fn)}
        rows.append(r)
        cpu=r["MX-CPU(ms)"] if isinstance(r["MX-CPU(ms)"],float) else None
        gpu=r["MX-GPU(ms)"] if isinstance(r["MX-GPU(ms)"],float) else None
        pol=r["Polars(ms)"] if isinstance(r["Polars(ms)"],float) else None
        sp_c=f"  CPU {pol/cpu:.1f}x vs Polars" if (cpu and pol and cpu>0) else ""
        sp_g=f"  GPU {pol/gpu:.1f}x vs Polars" if (gpu and pol and gpu>0) else ""
        def display(value):
            return f"{value}ms" if isinstance(value, (int, float)) else str(value)
        print(f"  cpu={display(r['MX-CPU(ms)'])}  gpu={display(r['MX-GPU(ms)'])}  polars={display(r['Polars(ms)'])}  pandas={display(r['Pandas(ms)'])}{sp_c}{sp_g}")

        clear_cache()
        gc.collect()

    specs = {
        1:("Filter+8Agg",lambda:run_q1_mxframe(data["li"],"cpu"),lambda:run_q1_mxframe(data["li"],"gpu"),lambda:run_q1_polars(data["li"]),lambda:run_q1_pandas(data["li"])),
        2:("Min Cost Supplier",lambda:run_q2_mxframe(*data[2],"cpu"),lambda:run_q2_mxframe(*data[2],"gpu"),lambda:run_q2_polars(*data[2]),lambda:run_q2_pandas(*data[2])),
        3:("3-Table Join+Agg",lambda:run_q3_mxframe(*data[3],"cpu"),lambda:run_q3_mxframe(*data[3],"gpu"),lambda:run_q3_polars(*data[3]),lambda:run_q3_pandas(*data[3])),
        4:("Order Priority",lambda:run_q4_mxframe(*data[4],"cpu"),lambda:run_q4_mxframe(*data[4],"gpu"),lambda:run_q4_polars(*data[4]),lambda:run_q4_pandas(*data[4])),
        5:("Multi-Join+GroupBy",lambda:run_q5_mxframe(*data[5],"cpu"),lambda:run_q5_mxframe(*data[5],"gpu"),lambda:run_q5_polars(*data[5]),lambda:run_q5_pandas(*data[5])),
        6:("Masked GlobalAgg",lambda:run_q6_mxframe(data["li"],"cpu"),lambda:run_q6_mxframe(data["li"],"gpu"),lambda:run_q6_polars(data["li"]),lambda:run_q6_pandas(data["li"])),
        7:("Shipping Volume",lambda:run_q7_mxframe(*data[7],"cpu"),lambda:run_q7_mxframe(*data[7],"gpu"),lambda:run_q7_polars(*data[7]),lambda:run_q7_pandas(*data[7])),
        8:("Market Share",lambda:run_q8_mxframe(*data[8],"cpu"),lambda:run_q8_mxframe(*data[8],"gpu"),lambda:run_q8_polars(*data[8]),lambda:run_q8_pandas(*data[8])),
        9:("Product Profit",lambda:run_q9_mxframe(*data[9],"cpu"),lambda:run_q9_mxframe(*data[9],"gpu"),lambda:run_q9_polars(*data[9]),lambda:run_q9_pandas(*data[9])),
        10:("Customer Revenue",lambda:run_q10_mxframe(*data[10],"cpu"),lambda:run_q10_mxframe(*data[10],"gpu"),lambda:run_q10_polars(*data[10]),lambda:run_q10_pandas(*data[10])),
        11:("Important Stock",lambda:run_q11_mxframe(*data[11],"cpu"),lambda:run_q11_mxframe(*data[11],"gpu"),lambda:run_q11_polars(*data[11]),lambda:run_q11_pandas(*data[11])),
        12:("2-Table Join+Agg",lambda:run_q12_mxframe(*data[12],"cpu"),lambda:run_q12_mxframe(*data[12],"gpu"),lambda:run_q12_polars(*data[12]),lambda:run_q12_pandas(*data[12])),
        13:("Customer Distrib",lambda:run_q13_mxframe(*data[13],"cpu"),lambda:run_q13_mxframe(*data[13],"gpu"),lambda:run_q13_polars(*data[13]),lambda:run_q13_pandas(*data[13])),
        14:("Promo Revenue",lambda:run_q14_mxframe(*data[14],"cpu"),lambda:run_q14_mxframe(*data[14],"gpu"),lambda:run_q14_polars(*data[14]),lambda:run_q14_pandas(*data[14])),
        15:("Top Supplier Rev",lambda:run_q15_mxframe(*data[15],"cpu"),lambda:run_q15_mxframe(*data[15],"gpu"),lambda:run_q15_polars(*data[15]),lambda:run_q15_pandas(*data[15])),
        16:("Part/Supplier Rel",lambda:run_q16_mxframe(*data[16],"cpu"),lambda:run_q16_mxframe(*data[16],"gpu"),lambda:run_q16_polars(*data[16]),lambda:run_q16_pandas(*data[16])),
        17:("Small Qty Order",lambda:run_q17_mxframe(*data[17],"cpu"),lambda:run_q17_mxframe(*data[17],"gpu"),lambda:run_q17_polars(*data[17]),lambda:run_q17_pandas(*data[17])),
        18:("Large Volume Cust",lambda:run_q18_mxframe(*data[18],"cpu"),lambda:run_q18_mxframe(*data[18],"gpu"),lambda:run_q18_polars(*data[18]),lambda:run_q18_pandas(*data[18])),
        19:("Discounted Revenue",lambda:run_q19_mxframe(*data[19],"cpu"),lambda:run_q19_mxframe(*data[19],"gpu"),lambda:run_q19_polars(*data[19]),lambda:run_q19_pandas(*data[19])),
        20:("Potential Part Promo",lambda:run_q20_mxframe(*data[20],"cpu"),lambda:run_q20_mxframe(*data[20],"gpu"),lambda:run_q20_polars(*data[20]),lambda:run_q20_pandas(*data[20])),
        21:("Suppliers Who Kept",lambda:run_q21_mxframe(*data[21],"cpu"),lambda:run_q21_mxframe(*data[21],"gpu"),lambda:run_q21_polars(*data[21]),lambda:run_q21_pandas(*data[21])),
        22:("Global Sales Oppty",lambda:run_q22_mxframe(*data[22],"cpu"),lambda:run_q22_mxframe(*data[22],"gpu"),lambda:run_q22_polars(*data[22]),lambda:run_q22_pandas(*data[22])),
    }
    for query in sorted(selected):
        row(f"Q{query}", *specs[query])

    df = pd.DataFrame(rows).set_index("Q")
    print(f"\n{'='*75}\nResults: {N:,} rows — warm median of 3 runs (ms)\n{'='*75}")
    if rapids_only:
        df = pd.DataFrame(rows).set_index("Q")
        print(df.to_string())
        df.to_csv(output)
        print(f"\nSaved: {output}")
        return df
    if engine is not None:
        df = pd.DataFrame(rows).set_index("Q")
        print(df.to_string())
        df.to_csv(output)
        print(f"\nSaved: {output}")
        return df
    pd.set_option('display.max_columns',None); pd.set_option('display.width',120)
    print(df[["Description","MX-CPU(ms)","MX-GPU(ms)","Polars(ms)","Pandas(ms)"]].to_string())
    csv_path = output or Path(__file__).with_name(f"bench_results_{N//1_000_000}M.csv")
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")
    return df

def run_batched(N, scale, batch_size, rapids, query_timeout):
    script = Path(__file__).resolve()
    frames = []
    rapids_frames = []
    rapids_available = importlib.util.find_spec("cudf") is not None
    if rapids and not rapids_available:
        print("RAPIDS cuDF: unavailable (cudf is not installed); recording N/A")
    for start in range(0, len(ALL_QUERIES), batch_size):
        batch = ALL_QUERIES[start:start + batch_size]
        print(f"\nStarting isolated batch Q{batch[0]}-Q{batch[-1]}")
        for query in batch:
            query_frames = []
            for engine, column in (
                ("mx-cpu", "MX-CPU(ms)"), ("mx-gpu", "MX-GPU(ms)"),
                ("polars", "Polars(ms)"), ("pandas", "Pandas(ms)"),
            ):
                output = script.with_name(f".bench_{N}_{query}_{engine}.csv")
                command = [sys.executable, str(script), "--worker", "--rows", str(N),
                           "--scale", str(scale), "--queries", str(query),
                           "--engine", engine, "--output", str(output)]
                try:
                    subprocess.run(command, check=True, timeout=query_timeout)
                    engine_frame = pd.read_csv(output).set_index(["Q", "Description"])
                except subprocess.TimeoutExpired:
                    print(f"Q{query} {engine} timed out after {query_timeout}s; continuing")
                    engine_frame = pd.DataFrame([{
                        "Q": f"Q{query}", "Description": DESCRIPTIONS[query],
                        column: "TIMEOUT",
                    }]).set_index(["Q", "Description"])
                finally:
                    output.unlink(missing_ok=True)
                query_frames.append(engine_frame)
            frames.append(pd.concat(query_frames, axis=1).reset_index())

            if rapids and rapids_available:
                rapids_output = script.with_name(f".bench_rapids_{N}_{query}.csv")
                rapids_command = [sys.executable, "-m", "cudf.pandas", str(script),
                                  "--worker", "--rapids-worker", "--rows", str(N),
                                  "--scale", str(scale), "--queries", str(query),
                                  "--output", str(rapids_output)]
                try:
                    subprocess.run(rapids_command, check=True, timeout=query_timeout)
                    rapids_frames.append(pd.read_csv(rapids_output))
                except subprocess.TimeoutExpired:
                    print(f"RAPIDS Q{query} timed out after {query_timeout}s; continuing")
                    rapids_frames.append(pd.DataFrame([{
                        "Q": f"Q{query}", "RAPIDS-cuDF(ms)": "TIMEOUT",
                    }]))
                finally:
                    rapids_output.unlink(missing_ok=True)

    result = pd.concat(frames, ignore_index=True)
    result["query_number"] = result["Q"].str.removeprefix("Q").astype(int)
    result = result.sort_values("query_number").drop(columns="query_number").set_index("Q")
    if rapids:
        if rapids_frames:
            rapids_result = pd.concat(rapids_frames, ignore_index=True).set_index("Q")
            result = result.join(rapids_result)
        else:
            result["RAPIDS-cuDF(ms)"] = "N/A"
    final = script.with_name(f"bench_results_{N//1_000_000}M.csv")
    result.to_csv(final)
    print(f"\nMerged all 22 queries: {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--10m", action="store_true", dest="run10m")
    ap.add_argument("--only-10m", action="store_true",
                    help="benchmark 10M rows without repeating the 1M run")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--query-timeout", type=int, default=300,
                    help="maximum seconds for one isolated query worker")
    ap.add_argument("--rapids", action="store_true",
                    help="benchmark Pandas query code through the official cudf.pandas accelerator")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rapids-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--engine", choices=("mx-cpu", "mx-gpu", "polars", "pandas"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--rows", type=int, default=1_000_000, help=argparse.SUPPRESS)
    ap.add_argument("--scale", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--queries", type=parse_queries, default=ALL_QUERIES, help=argparse.SUPPRESS)
    ap.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        bench_scale(args.rows, s=args.scale, selected=args.queries, output=args.output,
                    rapids_only=args.rapids_worker, engine=args.engine)
    else:
        if not args.only_10m:
            run_batched(1_000_000, 1, args.batch_size, args.rapids, args.query_timeout)
        if args.run10m or args.only_10m:
            run_batched(10_000_000, 10, args.batch_size, args.rapids, args.query_timeout)
    print("\n\nAll done.")
