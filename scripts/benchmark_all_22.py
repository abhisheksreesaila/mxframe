#!/usr/bin/env python3
"""
benchmark_all_22.py  --  MXFrame CPU | MXFrame GPU | Polars | Pandas
All 22 TPC-H queries. One-shot timing at 1M and 10M rows.
"""
import sys, time
sys.path.insert(0, '/home/ablearn/mxdf_v2')
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
    run_q1_mxframe, run_q1_pandas, run_q1_polars,
    run_q6_mxframe, run_q6_pandas, run_q6_polars,
    run_q3_mxframe, run_q3_pandas, run_q3_polars,
    run_q12_mxframe, run_q12_pandas, run_q12_polars,
    run_q14_mxframe, run_q14_pandas, run_q14_polars,
    run_q5_mxframe, run_q5_pandas, run_q5_polars,
    run_q10_mxframe, run_q10_pandas, run_q10_polars,
    run_q7_mxframe, run_q7_pandas, run_q7_polars,
    run_q8_mxframe, run_q8_pandas, run_q8_polars,
    run_q13_mxframe, run_q13_pandas, run_q13_polars,
    run_q19_mxframe, run_q19_pandas, run_q19_polars,
    run_q4_mxframe, run_q4_pandas, run_q4_polars,
    run_q9_mxframe, run_q9_pandas, run_q9_polars,
    run_q11_mxframe, run_q11_pandas, run_q11_polars,
    run_q18_mxframe, run_q18_pandas, run_q18_polars,
    run_q16_mxframe, run_q16_pandas, run_q16_polars,
    run_q17_mxframe, run_q17_pandas, run_q17_polars,
    run_q2_mxframe, run_q2_pandas, run_q2_polars,
    run_q15_mxframe, run_q15_pandas, run_q15_polars,
    run_q20_mxframe, run_q20_pandas, run_q20_polars,
    run_q21_mxframe, run_q21_pandas, run_q21_polars,
    run_q22_mxframe, run_q22_pandas, run_q22_polars,
)


def t(fn):
    t0 = time.perf_counter()
    try:
        fn()
        return round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        return f"ERR:{str(e)[:40]}"


def bench_scale(N, s=1):
    print(f"\n{'='*75}")
    print(f"  Benchmarking at {N:,} rows  (scale factor s={s})")
    print(f"{'='*75}")
    print("  Generating data...", end="", flush=True)

    li   = make_lineitem(N)
    c3, o3, l3   = make_tpch_q3_tables(n_customers=15000*s, n_orders=150000*s, n_lineitem=600000*s)
    o12, l12     = make_tpch_q12_tables(n_orders=50000*s, n_lineitem=300000*s)
    p14, l14     = make_tpch_q14_tables(n_parts=20000*s, n_lineitem=200000*s)
    na5, c5, o5, l5  = make_tpch_q5_tables(n_customers=15000*s, n_orders=100000*s, n_lineitem=400000*s)
    na10, c10, o10, l10 = make_tpch_q10_tables(n_customers=15000*s, n_orders=150000*s, n_lineitem=600000*s)
    na7, s7, c7, o7, l7 = make_tpch_q7_tables(n_sup=2000*s, n_cust=10000*s, n_orders=80000*s, n_li=200000*s)
    na8, r8, p8, c8, o8, l8 = make_tpch_q8_tables(n_parts=5000*s, n_cust=10000*s, n_orders=80000*s, n_li=200000*s)
    c13, o13     = make_tpch_q13_tables(n_customers=150000*s, n_orders=600000*s)
    p19, l19     = make_tpch_q19_tables(n_parts=20000*s, n_lineitem=200000*s)
    o4, l4       = make_tpch_q4_tables(n_orders=150000*s, n_lineitem=600000*s)
    na9, s9, ps9, p9, o9, l9 = make_tpch_q9_tables(n_parts=5000*s, n_suppliers=2000*s, n_orders=80000*s, n_lineitem=200000*s)
    na11, s11, ps11 = make_tpch_q11_tables()
    c18, o18, l18  = make_tpch_q18_tables(n_customer=15000*s, n_orders=60000*s, n_lineitem=300000*s)
    p16, s16, ps16 = make_tpch_q16_tables()
    p17, l17       = make_tpch_q17_tables(n_parts=5000*s, n_lineitem=200000*s)
    na2, r2, p2, s2, ps2 = make_tpch_q2_tables()
    s15, l15       = make_tpch_q15_tables(n_suppliers=2000*s, n_lineitem=200000*s)
    na20, p20, s20, ps20, l20 = make_tpch_q20_tables(n_parts=5000*s, n_suppliers=2000*s, n_lineitem=100000*s)
    na21, s21, o21, l21 = make_tpch_q21_tables(n_suppliers=2000*s, n_orders=60000*s, n_lineitem=200000*s)
    c22, o22       = make_tpch_q22_tables(n_customers=150000*s, n_orders=400000*s)
    print(" done")

    rows = []

    def row(qname, desc, cpu_fn, gpu_fn, pl_fn, pd_fn):
        print(f"  {qname:<5}", end="", flush=True)
        r = {"Q": qname, "Description": desc,
             "MX-CPU(ms)": t(cpu_fn),
             "MX-GPU(ms)": t(gpu_fn),
             "Polars(ms)": t(pl_fn),
             "Pandas(ms)": t(pd_fn)}
        rows.append(r)
        cpu = r["MX-CPU(ms)"] if isinstance(r["MX-CPU(ms)"], float) else None
        gpu = r["MX-GPU(ms)"] if isinstance(r["MX-GPU(ms)"], float) else None
        pol = r["Polars(ms)"] if isinstance(r["Polars(ms)"], float) else None
        sp_c = f"  CPU {pol/cpu:.1f}x vs Polars" if (cpu and pol and cpu>0) else ""
        sp_g = f"  GPU {pol/gpu:.1f}x vs Polars" if (gpu and pol and gpu>0) else ""
        print(f"  cpu={r['MX-CPU(ms)']}ms  gpu={r['MX-GPU(ms)']}ms  polars={r['Polars(ms)']}ms  pandas={r['Pandas(ms)']}ms{sp_c}{sp_g}")

    row("Q1",  "Filter+8Agg",          lambda: run_q1_mxframe(li,"cpu"),    lambda: run_q1_mxframe(li,"gpu"),    lambda: run_q1_polars(li),  lambda: run_q1_pandas(li))
    row("Q6",  "Masked GlobalAgg",      lambda: run_q6_mxframe(li,"cpu"),    lambda: run_q6_mxframe(li,"gpu"),    lambda: run_q6_polars(li),  lambda: run_q6_pandas(li))
    row("Q3",  "3-Table Join+Agg",      lambda: run_q3_mxframe(c3,o3,l3,"cpu"),  lambda: run_q3_mxframe(c3,o3,l3,"gpu"),  lambda: run_q3_polars(c3,o3,l3),  lambda: run_q3_pandas(c3,o3,l3))
    row("Q12", "2-Table Join+Agg",      lambda: run_q12_mxframe(o12,l12,"cpu"),  lambda: run_q12_mxframe(o12,l12,"gpu"),  lambda: run_q12_polars(o12,l12),  lambda: run_q12_pandas(o12,l12))
    row("Q14", "Promo Revenue",         lambda: run_q14_mxframe(p14,l14,"cpu"),  lambda: run_q14_mxframe(p14,l14,"gpu"),  lambda: run_q14_polars(p14,l14),  lambda: run_q14_pandas(p14,l14))
    row("Q5",  "Multi-Join+GroupBy",    lambda: run_q5_mxframe(na5,c5,o5,l5,"cpu"),  lambda: run_q5_mxframe(na5,c5,o5,l5,"gpu"),  lambda: run_q5_polars(na5,c5,o5,l5),  lambda: run_q5_pandas(na5,c5,o5,l5))
    row("Q10", "Customer Revenue",      lambda: run_q10_mxframe(na10,c10,o10,l10,"cpu"), lambda: run_q10_mxframe(na10,c10,o10,l10,"gpu"), lambda: run_q10_polars(na10,c10,o10,l10), lambda: run_q10_pandas(na10,c10,o10,l10))
    row("Q7",  "Shipping Volume",       lambda: run_q7_mxframe(na7,s7,c7,o7,l7,"cpu"),   lambda: run_q7_mxframe(na7,s7,c7,o7,l7,"gpu"),   lambda: run_q7_polars(na7,s7,c7,o7,l7),   lambda: run_q7_pandas(na7,s7,c7,o7,l7))
    row("Q8",  "Market Share",          lambda: run_q8_mxframe(na8,r8,p8,c8,o8,l8,"cpu"),lambda: run_q8_mxframe(na8,r8,p8,c8,o8,l8,"gpu"),lambda: run_q8_polars(na8,r8,p8,c8,o8,l8),lambda: run_q8_pandas(na8,r8,p8,c8,o8,l8))
    row("Q13", "Customer Distrib",      lambda: run_q13_mxframe(c13,o13,"cpu"), lambda: run_q13_mxframe(c13,o13,"gpu"), lambda: run_q13_polars(c13,o13), lambda: run_q13_pandas(c13,o13))
    row("Q19", "Discounted Revenue",    lambda: run_q19_mxframe(p19,l19,"cpu"), lambda: run_q19_mxframe(p19,l19,"gpu"), lambda: run_q19_polars(p19,l19), lambda: run_q19_pandas(p19,l19))
    row("Q4",  "Order Priority",        lambda: run_q4_mxframe(o4,l4,"cpu"),   lambda: run_q4_mxframe(o4,l4,"gpu"),   lambda: run_q4_polars(o4,l4),   lambda: run_q4_pandas(o4,l4))
    row("Q9",  "Product Profit",        lambda: run_q9_mxframe(na9,s9,ps9,p9,o9,l9,"cpu"), lambda: run_q9_mxframe(na9,s9,ps9,p9,o9,l9,"gpu"), lambda: run_q9_polars(na9,s9,ps9,p9,o9,l9), lambda: run_q9_pandas(na9,s9,ps9,p9,o9,l9))
    row("Q11", "Important Stock",       lambda: run_q11_mxframe(na11,s11,ps11,"cpu"), lambda: run_q11_mxframe(na11,s11,ps11,"gpu"), lambda: run_q11_polars(na11,s11,ps11), lambda: run_q11_pandas(na11,s11,ps11))
    row("Q18", "Large Volume Cust",     lambda: run_q18_mxframe(c18,o18,l18,"cpu"), lambda: run_q18_mxframe(c18,o18,l18,"gpu"), lambda: run_q18_polars(c18,o18,l18), lambda: run_q18_pandas(c18,o18,l18))
    row("Q16", "Part/Supplier Rel",     lambda: run_q16_mxframe(p16,s16,ps16,"cpu"), lambda: run_q16_mxframe(p16,s16,ps16,"gpu"), lambda: run_q16_polars(p16,s16,ps16), lambda: run_q16_pandas(p16,s16,ps16))
    row("Q17", "Small Qty Order",       lambda: run_q17_mxframe(p17,l17,"cpu"), lambda: run_q17_mxframe(p17,l17,"gpu"), lambda: run_q17_polars(p17,l17), lambda: run_q17_pandas(p17,l17))
    row("Q2",  "Min Cost Supplier",     lambda: run_q2_mxframe(na2,r2,p2,s2,ps2,"cpu"), lambda: run_q2_mxframe(na2,r2,p2,s2,ps2,"gpu"), lambda: run_q2_polars(na2,r2,p2,s2,ps2), lambda: run_q2_pandas(na2,r2,p2,s2,ps2))
    row("Q15", "Top Supplier Revenue",  lambda: run_q15_mxframe(s15,l15,"cpu"), lambda: run_q15_mxframe(s15,l15,"gpu"), lambda: run_q15_polars(s15,l15), lambda: run_q15_pandas(s15,l15))
    row("Q20", "Potential Part Promo",  lambda: run_q20_mxframe(na20,p20,s20,ps20,l20,"cpu"), lambda: run_q20_mxframe(na20,p20,s20,ps20,l20,"gpu"), lambda: run_q20_polars(na20,p20,s20,ps20,l20), lambda: run_q20_pandas(na20,p20,s20,ps20,l20))
    row("Q21", "Suppliers Who Kept",    lambda: run_q21_mxframe(na21,s21,o21,l21,"cpu"), lambda: run_q21_mxframe(na21,s21,o21,l21,"gpu"), lambda: run_q21_polars(na21,s21,o21,l21), lambda: run_q21_pandas(na21,s21,o21,l21))
    row("Q22", "Global Sales Oppty",    lambda: run_q22_mxframe(c22,o22,"cpu"), lambda: run_q22_mxframe(c22,o22,"gpu"), lambda: run_q22_polars(c22,o22), lambda: run_q22_pandas(c22,o22))

    df = pd.DataFrame(rows).set_index("Q")
    print(f"\n{'='*75}")
    print(f"Results: {N:,} rows — one-shot timing (ms)")
    print(f"{'='*75}")
    pd.set_option('display.max_columns', None); pd.set_option('display.width', 120)
    print(df[["Description","MX-CPU(ms)","MX-GPU(ms)","Polars(ms)","Pandas(ms)"]].to_string())
    csv_path = f"/home/ablearn/mxdf_v2/scripts/bench_results_{N//1_000_000}M.csv"
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")
    return df


if __name__ == "__main__":
    df1 = bench_scale(1_000_000, s=1)
    df10 = bench_scale(10_000_000, s=10)
    print("\n\nAll done.")
