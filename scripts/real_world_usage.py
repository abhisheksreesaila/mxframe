#!/usr/bin/env python3
"""
MXFrame — Real-World Usage Examples
====================================
This script shows how a regular developer would use MXFrame for typical
data analysis tasks: NOT a benchmark — just clean, readable API usage.

Run with:
  pixi run python scripts/real_world_usage.py

Covers:
  1. Basic filter, select, sort, limit
  2. GroupBy + aggregation
  3. Multi-table join
  4. Date filtering (year extraction)
  5. HAVING filter (post-aggregation)
  6. Left join (preserve all rows on one side)
  7. Complex OR predicates + isin
  8. CASE WHEN / conditional column
  9. Window-like: top-N per group (sort + limit)
 10. Chained operations (real analysis pipeline)
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from mxframe import LazyFrame, col, lit, when
from mxframe.lazy_frame import Scan
from mxframe.custom_ops import clear_cache

# Clear any stale GPU/model caches at startup
clear_cache()


# ── Helpers ──────────────────────────────────────────────────────────────────
def hr(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def show(result, label: str = None, limit: int = 8) -> None:
    df = result.to_pandas() if isinstance(result, pa.Table) else result
    if label:
        print(f"\n  [{label}]")
    print(df.head(limit).to_string(index=False))
    if len(df) > limit:
        print(f"  … {len(df) - limit} more rows …")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Generate some realistic-looking data
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

REGIONS    = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Clothing", "Books", "Home", "Sports"]
STATUSES   = ["completed", "pending", "cancelled", "refunded"]

N = 50_000  # orders

orders = pa.table({
    "order_id":    pa.array(np.arange(1, N + 1, dtype=np.int32)),
    "customer_id": pa.array(rng.integers(1, 5001, N, dtype=np.int32)),
    "region":      pa.array([REGIONS[i] for i in rng.integers(0, len(REGIONS), N)]),
    "category":    pa.array([CATEGORIES[i] for i in rng.integers(0, len(CATEGORIES), N)]),
    "status":      pa.array([STATUSES[i] for i in rng.integers(0, len(STATUSES), N)]),
    "amount":      pa.array(rng.uniform(5.0, 2000.0, N).astype(np.float32)),
    "quantity":    pa.array(rng.integers(1, 21, N, dtype=np.int32)),
    "order_date":  pa.array(rng.integers(20190101, 20240101, N, dtype=np.int32)),
})

customers = pa.table({
    "customer_id":  pa.array(np.arange(1, 5001, dtype=np.int32)),
    "customer_name":pa.array([f"Customer #{i}" for i in range(1, 5001)]),
    "tier":         pa.array(rng.choice(["Bronze", "Silver", "Gold", "Platinum"],
                                         size=5000).tolist()),
    "credit_limit": pa.array(rng.uniform(500.0, 50000.0, 5000).astype(np.float32)),
})

print("=" * 60)
print("  MXFrame — Real-World Usage Demo")
print("=" * 60)
print(f"  Orders:    {orders.num_rows:,} rows")
print(f"  Customers: {customers.num_rows:,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic filter + sort + limit   (SQL: SELECT … WHERE … ORDER BY … LIMIT)
# ─────────────────────────────────────────────────────────────────────────────
hr("1. Filter + Sort + Limit  (top 5 large completed orders)")

result = (
    LazyFrame(Scan(orders))
    .filter(
        (col("status") == lit("completed")) &
        (col("amount") > lit(1500.0))
    )
    .sort(col("amount"), descending=True)
    .limit(5)
    .compute()
)
show(result, "Top 5 completed orders > $1500")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GroupBy + multiple aggregations   (SQL: GROUP BY … aggregate functions)
# ─────────────────────────────────────────────────────────────────────────────
hr("2. GroupBy + Aggregations  (revenue by category and region)")

result = (
    LazyFrame(Scan(orders))
    .filter(col("status") == lit("completed"))
    .groupby("category", "region")
    .agg(
        col("amount").sum().alias("total_revenue"),
        col("amount").mean().alias("avg_order"),
        col("order_id").count().alias("num_orders"),
    )
    .sort(col("total_revenue"), descending=True)
    .compute()
)
show(result, "Revenue by category & region")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inner join   (SQL: JOIN customers ON customer_id)
# ─────────────────────────────────────────────────────────────────────────────
hr("3. Inner Join  (orders + customer tier)")

result = (
    LazyFrame(Scan(orders))
    .filter(col("status") == lit("completed"))
    .join(LazyFrame(Scan(customers)), on="customer_id")
    .groupby("tier")
    .agg(
        col("amount").sum().alias("total_revenue"),
        col("order_id").count().alias("num_orders"),
    )
    .sort(col("total_revenue"), descending=True)
    .compute()
)
show(result, "Revenue by customer tier")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Date filtering — year extraction
# ─────────────────────────────────────────────────────────────────────────────
hr("4. Date Filtering  (filter orders in 2022 + group by year)")

# Filter using year() expression
orders_2022 = (
    LazyFrame(Scan(orders))
    .filter(col("order_date").year() == lit(2022))
    .compute()
)
print(f"  Orders in 2022: {orders_2022.num_rows:,}")

# Group by year (pre-compute year column, then groupby)
year_col = pc.cast(
    pc.divide(orders.column("order_date"), pa.scalar(10000, type=pa.int32())),
    pa.int32()
)
orders_with_year = orders.append_column("order_year", year_col)

result = (
    LazyFrame(Scan(orders_with_year))
    .filter(col("status") == lit("completed"))
    .groupby("order_year")
    .agg(
        col("amount").sum().alias("annual_revenue"),
        col("order_id").count().alias("num_orders"),
    )
    .sort(col("order_year"))
    .compute()
)
show(result, "Annual revenue by year (completed orders)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HAVING filter  (SQL: GROUP BY … HAVING aggregate_condition)
# ─────────────────────────────────────────────────────────────────────────────
hr("5. HAVING Filter  (categories with > $500K total revenue)")

result = (
    LazyFrame(Scan(orders))
    .filter(col("status") == lit("completed"))
    .groupby("category")
    .agg(
        col("amount").sum().alias("total_revenue"),
        col("order_id").count().alias("num_orders"),
    )
    .filter(col("total_revenue") > lit(500_000.0))   # ← HAVING
    .sort(col("total_revenue"), descending=True)
    .compute()
)
show(result, "Categories with > $500K revenue")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Left Join  (SQL: LEFT OUTER JOIN — keep all customers, even inactive ones)
# ─────────────────────────────────────────────────────────────────────────────
hr("6. Left Join  (all customers, even those with no completed orders)")

# Get only completed orders
completed_by_cust = (
    LazyFrame(Scan(orders))
    .filter(col("status") == lit("completed"))
    .groupby("customer_id")
    .agg(
        col("amount").sum().alias("total_spent"),
        col("order_id").count().alias("order_count"),
    )
    .compute()
)

# Left join: all customers, null for those with no completed orders
result = (
    LazyFrame(Scan(customers))
    .join(
        LazyFrame(Scan(completed_by_cust)),
        on="customer_id",
        how="left"
    )
    .limit(10)
    .compute()
)
print(f"  All customers + their spending (showing 10):")
show(result)
# Count customers with no completed orders (null total_spent)
null_count = sum(1 for v in result.column("total_spent").to_pylist() if v is None)
print(f"  Customers with no completed orders in sample: {null_count}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Complex OR predicates + isin   (SQL: WHERE … OR … OR …)
# ─────────────────────────────────────────────────────────────────────────────
hr("7. Complex OR + isin  (premium orders: high value in Electronics/Sports OR large qty)")

result = (
    LazyFrame(Scan(orders))
    .filter(
        col("status").isin(["completed", "pending"])  # only active orders
    )
    .filter(
        # High-value electronics/sports OR bulk orders in any category
        (col("category").isin(["Electronics", "Sports"]) & (col("amount") > lit(800.0))) |
        (col("quantity") >= lit(15))
    )
    .groupby("category")
    .agg(
        col("amount").sum().alias("premium_revenue"),
        col("order_id").count().alias("premium_orders"),
    )
    .sort(col("premium_revenue"), descending=True)
    .compute()
)
show(result, "Premium orders by category")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CASE WHEN / conditional column   (classify orders by amount)
# ─────────────────────────────────────────────────────────────────────────────
hr("8. CASE WHEN  (revenue that qualifies for expedited fulfillment)")

# Expedited = orders over $500 in North/East; or any order over $1000
# Using when(condition, then, else) for a numeric flag
result = (
    LazyFrame(Scan(orders))
    .filter(col("status") == lit("completed"))
    .groupby("region")
    .agg(
        col("amount").sum().alias("total_revenue"),
        # Expedited portion: amount if > 500, else 0
        when(col("amount") > lit(500.0), col("amount"), lit(0.0))
            .sum().alias("expedited_revenue"),
    )
    .sort(col("total_revenue"), descending=True)
    .compute()
)
show(result, "Total vs expedited revenue by region")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Distinct   (unique values)
# ─────────────────────────────────────────────────────────────────────────────
hr("9. Distinct  (unique customer tiers ordering Electronics)")

result = (
    LazyFrame(Scan(orders))
    .filter(col("category") == lit("Electronics"))
    .join(LazyFrame(Scan(customers)), on="customer_id")
    .distinct("tier")
    .sort(col("tier"))
    .compute()
)
show(result, "Tiers ordering Electronics")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full Analysis Pipeline  (realistic chained query)
# ─────────────────────────────────────────────────────────────────────────────
hr("10. Full Pipeline  (top 5 Gold/Platinum customers by 2022 revenue, Electronics or Home)")

# Pre-compute year column
year_col = pc.cast(
    pc.divide(orders.column("order_date"), pa.scalar(10000, type=pa.int32())),
    pa.int32()
)
orders_y = orders.append_column("order_year", year_col)

result = (
    LazyFrame(Scan(orders_y))
    .filter(
        (col("status") == lit("completed")) &
        (col("order_year") == lit(2022)) &
        col("category").isin(["Electronics", "Home"])
    )
    .join(
        LazyFrame(Scan(customers))
        .filter(col("tier").isin(["Gold", "Platinum"])),
        on="customer_id",
    )
    .groupby("customer_id", "tier", "customer_name")
    .agg(
        col("amount").sum().alias("revenue_2022"),
        col("order_id").count().alias("orders_2022"),
    )
    .sort(col("revenue_2022"), descending=True)
    .limit(5)
    .compute()
)
show(result, "Top 5 Gold/Platinum customers (Electronics+Home, 2022)")


# ─────────────────────────────────────────────────────────────────────────────
# API Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  API Quick Reference")
print("=" * 60)
print("""
  LazyFrame(Scan(table))              — wrap a PyArrow table
    .filter(col("x") > lit(5.0))      — WHERE condition
    .filter(col("x").year() == lit(2022))  — date year filter
    .filter(col("x").isin(["a","b"])) — IN list
    .filter(cond1 | cond2)            — OR conditions
    .filter(~cond)                    — NOT condition
    .groupby("col1", "col2")          — GROUP BY
    .agg(col("x").sum().alias("s"),   — aggregate expressions
         col("y").mean().alias("m"),
         col("z").count().alias("c"))
    .filter(col("s") > lit(100.0))    — HAVING (after groupby)
    .sort(col("x"), descending=True)  — ORDER BY
    .limit(10)                        — LIMIT
    .distinct("col")                  — DISTINCT
    .select(col("a"), col("b").alias("c"))  — SELECT / rename
    .join(other_lf, on="key")         — INNER JOIN
    .join(other_lf, on="key", how="left")  — LEFT JOIN
    .compute(device="cpu")            — execute, return pa.Table
    .compute(device="gpu")            — GPU execution

  Expressions:
    col("name")                 — column reference
    lit(value)                  — literal value
    col("x").year()             — extract year from YYYYMMDD int
    col("x").startswith("P")    — string prefix filter
    when(cond, then, else_val)  — CASE WHEN expression
""")

print("Real-world usage demo complete!")
