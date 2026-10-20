# F-149: edits a small Parquet file (the pq_scan_probe schema) through a Frame over a scan at the edges of row groups and
# of blocks, queries it with the scan loop the arguments name, and prints one JSON line: the words, and per word the sum
# of cost and the row count. The edits are a fixed script an outside oracle replays (tests/test_mojo_parquet.py):
#   EDGES = rows 0, 1, 999, 1000, 1001, 4095, 4096, 4097, 13336, 13337, 13338, 26673, 26674, 26675, 40010
#   cost of EDGES[i] := 10_000_000 + i
#   region of EDGES[i] := "Edge" for even i
#   rows EDGES[i] removed for i % 4 == 1
# Usage: pq_edge_probe <path> <morsels: 0 off, 1 on, 2 auto> <block rows>
from std.sys import argv
from omxcore import Frame, Query, col, sum, count, scan_parquet, AUTO, MORSELS_AUTO, MORSELS_ON, MORSELS_OFF


def main() raises:
    var args = argv()
    var path = String(args[1])
    var mode = Int(String(args[2]))
    var block = Int(String(args[3]))
    var f = Frame(scan_parquet(path))
    var cost = f.column("cost")
    var region = f.column("region")
    var edges: List[Int] = [0, 1, 999, 1000, 1001, 4095, 4096, 4097, 13336, 13337, 13338, 26673, 26674, 26675, 40010]
    for i in range(len(edges)):
        f.set(cost, edges[i], Int64(10_000_000 + i))
        if i % 2 == 0:
            f.set_text(region, edges[i], "Edge")
    for i in range(len(edges)):
        if i % 4 == 1:
            f.remove(edges[i])
    var n = f.word_count(region)
    var m = MORSELS_ON if mode == 1 else MORSELS_OFF if mode == 0 else MORSELS_AUTO
    var r = Query(f).morsels(m).block_rows(block).group_by(col("region"), groups=n).agg(sum(col("cost")), count()).compute(AUTO)
    var words = String("")
    var sums = String("")
    var counts = String("")
    for g in range(n):
        words += ("" if g == 0 else ", ") + '"' + f.text_of(region, Int64(g)) + '"'
        sums += ("" if g == 0 else ", ") + String(r.values(0)[g])
        counts += ("" if g == 0 else ", ") + String(r.values(1)[g])
    print('{"words": [' + words + '], "sums": [' + sums + '], "counts": [' + counts + ']}')
