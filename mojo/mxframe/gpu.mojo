"""A device copy of a shared table (docs/mxframe-mojo-api.md section 7).

    var dt = DeviceTable(shared)                 # uploads every column once (a default DeviceContext, GPU 0)
    var dt = DeviceTable(shared, app_ctx)        # or on the app's own DeviceContext
    var r = Query(frame).on_device(dt).where(...).group_by(...).agg(...).compute(device=GPU)
    dt.release()                                 # the document closed: give the memory back

Why it looks like this:
- int64 throughout: the stand-in uploaded int32 and refused a wider value (a single amount over $21.4M in cents
  does not fit); here the columns go up exactly as the table holds them, validity bitmaps (Arrow layout) beside them.
- A `DeviceTable` is a handle like `SharedTable`: copies share one device copy, so every session holds the one the
  process uploaded. `release()` frees it for all of them (a query on a released table raises on `GPU` and runs on
  the CPU on `AUTO`).
- The app's `DeviceContext` is used as is: buffers are allocated on it and the query kernels are enqueued on it, so
  the app's own kernels (charts, point clouds) and MX frame's share one device, one memory pool and one stream. With
  the default constructor MX frame creates its own context on device 0.
- It is read-only: a session's edits never touch it (a query uploads the small bitmap of changed rows instead).
"""
from std.memory import ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer
from .table import SharedTable


def gpu_available() -> Bool:
    """Is there a GPU MX frame can use? (Creating a DeviceContext with none present crashes: ask first.)"""
    try:
        return DeviceContext.number_of_devices() > 0
    except:
        return False


struct _DeviceState(Movable):
    var ctx: DeviceContext
    var source: Int                                  # identity of the SharedTable this mirrors
    var rows: Int
    var live: Bool
    var bytes: Int
    var data: List[DeviceBuffer[DType.int64]]
    var valid: List[DeviceBuffer[DType.uint8]]       # only the columns that have nulls
    var data_addr: List[Int]
    var valid_addr: List[Int]                        # 0: the column has no nulls

    def __init__(out self, var ctx: DeviceContext, source: Int, rows: Int):
        self.ctx = ctx^
        self.source = source
        self.rows = rows
        self.live = True
        self.bytes = 0
        self.data = List[DeviceBuffer[DType.int64]]()
        self.valid = List[DeviceBuffer[DType.uint8]]()
        self.data_addr = List[Int]()
        self.valid_addr = List[Int]()


struct DeviceTable(ImplicitlyCopyable):
    """The columns of a `SharedTable` on the GPU, uploaded once and shared by every session that holds a copy."""
    var _s: ArcPointer[_DeviceState]

    def __init__(out self, table: SharedTable) raises:
        """Uploads `table` to the default GPU (device 0)."""
        if not gpu_available():
            raise Error("DeviceTable: no GPU found")
        self = Self(table, DeviceContext())

    def __init__(out self, table: SharedTable, ctx: DeviceContext) raises:
        """Uploads `table` on the app's `DeviceContext`: the app's buffers and these share one device."""
        self._s = ArcPointer(_DeviceState(ctx.copy(), Int(table._table[].cols.unsafe_ptr()), table.rows()))
        ref s = self._s[]
        var n = table.rows()
        for c in range(table.columns()):
            var buf = ctx.enqueue_create_buffer[DType.int64](max(n, 1))
            ctx.enqueue_copy(dst_buf=buf, src_ptr=table.values(c).unsafe_ptr())
            s.data_addr.append(Int(buf.unsafe_ptr()))
            s.data.append(buf^)
            s.bytes += 8 * max(n, 1)
            var nbytes = len(table._table[].cols[c].validity)
            if nbytes > 0:
                var vb = ctx.enqueue_create_buffer[DType.uint8](nbytes)
                ctx.enqueue_copy(dst_buf=vb, src_ptr=table._table[].cols[c].validity.unsafe_ptr())
                s.valid_addr.append(Int(vb.unsafe_ptr()))
                s.valid.append(vb^)
                s.bytes += nbytes
            else:
                s.valid_addr.append(0)
        ctx.synchronize()                            # the host columns may be freed or changed after this returns

    def context(self) -> DeviceContext:
        return self._s[].ctx.copy()

    def rows(self) -> Int:
        return self._s[].rows

    def bytes(self) -> Int:
        """Device memory held (0 after `release`)."""
        return self._s[].bytes if self._s[].live else 0

    def is_live(self) -> Bool:
        return self._s[].live

    def mirrors(self, table: SharedTable) -> Bool:
        """Was this made from `table` (the same shared columns, not merely an equal-looking table)?"""
        return self._s[].source == Int(table._table[].cols.unsafe_ptr())

    def data_address(self, c: Int) -> Int:
        return self._s[].data_addr[c]

    def validity_address(self, c: Int) -> Int:
        return self._s[].valid_addr[c]

    def release(self):
        """Frees the device memory for every copy of this handle (the document closed)."""
        ref s = self._s[]
        if not s.live:
            return
        s.data = List[DeviceBuffer[DType.int64]]()
        s.valid = List[DeviceBuffer[DType.uint8]]()
        s.data_addr = List[Int]()
        s.valid_addr = List[Int]()
        s.bytes = 0
        s.live = False
