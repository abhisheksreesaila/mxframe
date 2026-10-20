"""The jobs (F-175, job.mojo has the handle): every long piece of work an app starts, off its own thread.

    compute_job(query, device)               -> Job[QueryResult]    the builder's Query.compute
    sql_job(catalog, text, options)          -> Job[SqlResult]      sql(catalog, text)
    read_csv_job(path, options)              -> Job[CsvResult]      read_csv
    read_parquet_job(path)                   -> Job[ParquetResult]  read_parquet
    save_job(frame, path, ...)               -> Job[Int]            Frame.save (the rows written)
    write_parquet_job(table, path, ...)      -> Job[Int]            write_parquet

Each takes its inputs by value or copies them (the job owns what it works on): a `Query` is moved in, a `Catalog` and a `Frame`
are snapshotted when the job starts, so the caller may go on editing the frame or the catalog's sources; a `SharedTable` is
shared read only. The result of a job is what the plain call returns, value for value. Each also has a `*_then` form taking a
`JobDone` callback (job.mojo).
"""
from .job import Job, JobCtl, JobBody, JobDone, NoCallback, start_job_then
from .query import Query, QueryResult, AUTO


struct QueryJob(JobBody):
    comptime Out = QueryResult
    var query: Query
    var device: Int

    def __init__(out self, var query: Query, device: Int):
        self.query = query^
        self.device = device

    def run(mut self, ctl: JobCtl) raises -> QueryResult:
        return self.query.compute(self.device, ctl)


def compute_job[CB: JobDone = NoCallback](var query: Query, device: Int = AUTO, var done: CB = NoCallback()) -> Job[QueryResult]:
    """`query.compute(device)` on a thread of its own. Moves the query in; build it again for another run."""
    return start_job_then(QueryJob(query^, device), done^)
