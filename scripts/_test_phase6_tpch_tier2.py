import numpy as np
import pyarrow as pa
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mxframe import LazyFrame, col, lit, when
from mxframe.custom_ops import clear_cache

DEVICE = "cpu"


def reset():
    clear_cache()


def test_isin_filter():
    reset()
    tbl = pa.table({"mode": ["MAIL", "SHIP", "TRUCK", "MAIL"], "val": [1.0, 2.0, 3.0, 4.0]})
    r = (LazyFrame(tbl)
         .filter(col("mode").isin(["MAIL", "SHIP"]))
         .groupby()
         .agg(col("val").sum().alias("s"))
         .compute(device=DEVICE))
    assert float(r.column("s")[0].as_py()) == 7.0
    print("OK Test 1: isin() filter predicate")


def test_startswith_filter():
    reset()
    tbl = pa.table({"ptype": ["PROMO STEEL", "BRUSHED COPPER", "PROMO BRASS", "STANDARD STEEL"],
                    "val": [10.0, 20.0, 30.0, 40.0]})
    r = (LazyFrame(tbl)
         .filter(col("ptype").startswith("PROMO"))
         .groupby()
         .agg(col("val").sum().alias("s"))
         .compute(device=DEVICE))
    assert float(r.column("s")[0].as_py()) == 40.0
    print("OK Test 2: startswith() filter predicate")


def test_invert_isin():
    reset()
    tbl = pa.table({"mode": ["MAIL", "SHIP", "TRUCK", "AIR"], "val": [1.0, 2.0, 3.0, 4.0]})
    r = (LazyFrame(tbl)
         .filter(~col("mode").isin(["MAIL", "SHIP"]))
         .groupby()
         .agg(col("val").sum().alias("s"))
         .compute(device=DEVICE))
    assert float(r.column("s")[0].as_py()) == 7.0
    print("OK Test 3: ~isin (logical NOT) filter")


def test_ne_filter():
    reset()
    tbl = pa.table({"x": [1.0, 2.0, 3.0, 2.0], "y": [10.0, 20.0, 30.0, 40.0]})
    r = (LazyFrame(tbl)
         .filter(col("x") != lit(2.0))
         .groupby()
         .agg(col("y").sum().alias("s"))
         .compute(device=DEVICE))
    assert float(r.column("s")[0].as_py()) == 40.0
    print("OK Test 4: ne (!=) filter")


def test_case_when_numeric():
    reset()
    tbl = pa.table({"x": [1.0, 5.0, 3.0, 7.0]})
    r = (LazyFrame(tbl)
         .groupby()
         .agg(when(col("x") > lit(4.0), lit(1.0), lit(0.0)).sum().alias("high"))
         .compute(device=DEVICE))
    assert float(r.column("high")[0].as_py()) == 2.0
    print("OK Test 5: when() with numeric condition")


def test_case_when_isin():
    reset()
    tbl = pa.table({"priority": ["1-URGENT", "2-HIGH", "3-MEDIUM", "1-URGENT", "5-LOW"],
                    "c": [1.0, 1.0, 1.0, 1.0, 1.0]})
    r = (LazyFrame(tbl)
         .groupby()
         .agg(
             when(col("priority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0)).sum().alias("high"),
             when(~col("priority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0)).sum().alias("low"),
         )
         .compute(device=DEVICE))
    assert float(r.column("high")[0].as_py()) == 3.0, str(r)
    assert float(r.column("low")[0].as_py()) == 2.0, str(r)
    print("OK Test 6: when() with isin condition")


def test_case_when_startswith():
    reset()
    tbl = pa.table({"ptype": ["PROMO STEEL", "BRUSHED COPPER", "PROMO BRASS"],
                    "extendedprice": [200.0, 300.0, 100.0],
                    "discount": [0.1, 0.05, 0.2]})
    promo = when(col("ptype").startswith("PROMO"),
                 col("extendedprice") * (lit(1.0) - col("discount")),
                 lit(0.0))
    r = (LazyFrame(tbl)
         .groupby()
         .agg(promo.sum().alias("promo_rev"))
         .compute(device=DEVICE))
    got = float(r.column("promo_rev")[0].as_py())
    # 200*(1-0.1) + 100*(1-0.2) = 180 + 80 = 260
    assert abs(got - 260.0) < 0.5, f"got {got}"
    print("OK Test 7: when() with startswith condition")


def test_q12_style():
    reset()
    rng = np.random.default_rng(42)
    n_orders, n_li = 5000, 20000
    orders = pa.table({
        "o_orderkey": np.arange(1, n_orders + 1, dtype=np.int32),
        "o_orderpriority": rng.choice(
            ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"],
            size=n_orders).tolist(),
    })
    lineitem = pa.table({
        "l_orderkey": rng.integers(1, n_orders + 1, size=n_li, dtype=np.int32),
        "l_shipmode": rng.choice(["MAIL", "SHIP", "TRUCK", "AIR", "REG AIR"], size=n_li).tolist(),
        "l_commitdate": rng.integers(8000, 9000, size=n_li, dtype=np.int32),
        "l_receiptdate": rng.integers(8500, 9500, size=n_li, dtype=np.int32),
        "l_shipdate": rng.integers(7500, 8800, size=n_li, dtype=np.int32),
    })
    result = (
        LazyFrame(lineitem)
        .join(LazyFrame(orders), left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("l_shipmode").isin(["MAIL", "SHIP"]))
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .filter(col("l_shipdate") < col("l_commitdate"))
        .filter(col("l_receiptdate") >= lit(8760))
        .filter(col("l_receiptdate") < lit(9126))
        .groupby("l_shipmode")
        .agg(
            when(col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0))
                .sum().alias("high_line_count"),
            when(~col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0))
                .sum().alias("low_line_count"),
        )
        .sort(col("l_shipmode"))
        .compute(device=DEVICE)
    )
    try:
        import duckdb
        con = duckdb.connect()
        con.register("orders_tbl", orders)
        con.register("lineitem_tbl", lineitem)
        ref = con.execute("""
            SELECT l_shipmode,
                   SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH')
                            THEN 1 ELSE 0 END) high_line_count,
                   SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH')
                            THEN 1 ELSE 0 END) low_line_count
            FROM lineitem_tbl JOIN orders_tbl ON l_orderkey=o_orderkey
            WHERE l_shipmode IN ('MAIL','SHIP') AND l_commitdate<l_receiptdate
              AND l_shipdate<l_commitdate AND l_receiptdate>=8760 AND l_receiptdate<9126
            GROUP BY l_shipmode ORDER BY l_shipmode
        """).arrow().read_all()
        mx_modes = sorted([r.as_py() for r in result.column("l_shipmode")])
        ref_modes = sorted([r.as_py() for r in ref.column("l_shipmode")])
        assert mx_modes == ref_modes, f"Q12 modes: {mx_modes} vs {ref_modes}"
        for mode in mx_modes:
            mi = [r.as_py() for r in result.column("l_shipmode")].index(mode)
            ri = [r.as_py() for r in ref.column("l_shipmode")].index(mode)
            for agg in ("high_line_count", "low_line_count"):
                mv = float(result.column(agg)[mi].as_py())
                rv = float(ref.column(agg)[ri].as_py())
                assert abs(mv - rv) < 1.0, f"Q12 {mode} {agg}: MX={mv}, DuckDB={rv}"
        print("OK Test 8: Q12-style (join + isin + grouped case_when) -- matches DuckDB")
    except ImportError:
        assert result.num_rows == 2
        print("OK Test 8: Q12-style -- shape OK (no DuckDB)")


def test_q14_style():
    reset()
    rng = np.random.default_rng(99)
    n_parts, n_li = 2000, 10000
    part = pa.table({
        "p_partkey": np.arange(1, n_parts + 1, dtype=np.int32),
        "p_type": rng.choice(
            ["PROMO STEEL", "PROMO BRASS", "BRUSHED COPPER", "STANDARD STEEL", "ECONOMY TIN"],
            size=n_parts).tolist(),
    })
    lineitem = pa.table({
        "l_partkey": rng.integers(1, n_parts + 1, size=n_li, dtype=np.int32),
        "l_extendedprice": rng.uniform(900.0, 100000.0, size=n_li).astype(np.float32),
        "l_discount": rng.uniform(0.0, 0.1, size=n_li).astype(np.float32),
        "l_shipdate": rng.integers(9132, 9500, size=n_li, dtype=np.int32),
    })
    DATE_LO, DATE_HI = 9132, 9193
    promo_col = when(
        col("p_type").startswith("PROMO"),
        col("l_extendedprice") * (lit(1.0) - col("l_discount")),
        lit(0.0),
    )
    total_col = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
    result = (
        LazyFrame(lineitem)
        .join(LazyFrame(part), left_on="l_partkey", right_on="p_partkey")
        .filter(col("l_shipdate") >= lit(DATE_LO))
        .filter(col("l_shipdate") < lit(DATE_HI))
        .groupby()
        .agg(
            promo_col.sum().alias("promo_revenue"),
            total_col.sum().alias("total_revenue"),
        )
        .compute(device=DEVICE)
    )
    promo_rev = float(result.column("promo_revenue")[0].as_py())
    total_rev = float(result.column("total_revenue")[0].as_py())
    pct = 100.0 * promo_rev / total_rev if total_rev > 0 else 0.0
    try:
        import duckdb
        con = duckdb.connect()
        con.register("part_tbl", part)
        con.register("lineitem_tbl", lineitem)
        dql = (
            f"SELECT SUM(CASE WHEN p_type LIKE 'PROMO%' "
            f"              THEN l_extendedprice*(1-l_discount) ELSE 0 END) p, "
            f"       SUM(l_extendedprice*(1-l_discount)) t "
            f"FROM lineitem_tbl JOIN part_tbl ON l_partkey=p_partkey "
            f"WHERE l_shipdate>={DATE_LO} AND l_shipdate<{DATE_HI}"
        )
        ref = con.execute(dql).fetchone()
        ref_promo, ref_total = float(ref[0] or 0), float(ref[1] or 0)
        ref_pct = 100.0 * ref_promo / ref_total if ref_total > 0 else 0.0
        assert abs(pct - ref_pct) < 0.5, f"Q14: MX={pct:.4f}%, DuckDB={ref_pct:.4f}%"
        print(f"OK Test 9: Q14-style -- promo_revenue={pct:.2f}% (DuckDB={ref_pct:.2f}%)")
    except ImportError:
        assert 0.0 < pct < 100.0
        print(f"OK Test 9: Q14-style -- promo_revenue={pct:.2f}% (no DuckDB)")


if __name__ == "__main__":
    tests = [
        test_isin_filter, test_startswith_filter, test_invert_isin,
        test_ne_filter, test_case_when_numeric, test_case_when_isin,
        test_case_when_startswith, test_q12_style, test_q14_style,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {e}")
            sys.exit(1)
    print()
    print("All Phase 6 (TPC-H Tier 2) tests passed!")
