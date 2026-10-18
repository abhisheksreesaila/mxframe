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
- F-138b, where the GPU code lives: only on a GPU build. `HAS_GPU` is true on Linux (the target, as the app's D-56
  compiles its GPU path out of Apple builds); `-D MXFRAME_NO_GPU` turns it off on Linux too (to test the other side).
  Anywhere else `gpu_available()` is False, `DeviceTable(...)` raises "no GPU on this build", `Query.compute(GPU)`
  raises the same and AUTO runs on the CPU. Every use of `DeviceContext`, a buffer or `kernels.pipeline_gpu` sits in
  a `comptime if HAS_GPU` block (the imports stay; a module-level import is not code), so an Apple object refers to
  no `AsyncRT_Device*` symbol. The device memory hangs off the shared state by address (`_GpuMemory`) for that reason:
  a field of that type would put its destructor, and so the GPU runtime, into every build.
"""
from std.memory import ArcPointer
from std.memory.alloc import unsafe_alloc
from std.sys import is_defined, CompilationTarget
from max.gpu.host import DeviceContext, DeviceBuffer
from .table import SharedTable

comptime HAS_GPU = CompilationTarget.is_linux() and not is_defined["MXFRAME_NO_GPU"]()
comptime NO_GPU_MESSAGE = "no GPU on this build"


def gpu_available() -> Bool:
    """Is there a GPU MX frame can use? (Creating a DeviceContext with none present crashes: ask first.) Always False
    on a build without GPU code."""
    comptime if HAS_GPU:
        try:
            return DeviceContext.number_of_devices() > 0
        except:
            return False
    else:
        return False


struct _GpuMemory(Movable):
    """What lives on the device. Built and used only inside `comptime if HAS_GPU`."""
    var ctx: DeviceContext
    var data: List[DeviceBuffer[DType.int64]]
    var valid: List[DeviceBuffer[DType.uint8]]       # only the columns that have nulls

    def __init__(out self, var ctx: DeviceContext):
        self.ctx = ctx^
        self.data = List[DeviceBuffer[DType.int64]]()
        self.valid = List[DeviceBuffer[DType.uint8]]()


struct _DeviceState(Movable):
    var table: SharedTable                           # the table this mirrors, held so its identity cannot be reused
    var rows: Int
    var live: Bool
    var bytes: Int
    var data_addr: List[Int]
    var valid_addr: List[Int]                        # 0: the column has no nulls
    var gpu: Int                                     # address of the _GpuMemory (0: none, or released)

    def __init__(out self, table: SharedTable, rows: Int):
        self.table = table
        self.rows = rows
        self.live = True
        self.bytes = 0
        self.data_addr = List[Int]()
        self.valid_addr = List[Int]()
        self.gpu = 0

    def _free_gpu(mut self):
        comptime if HAS_GPU:
            if self.gpu != 0:
                var p = UnsafePointer[_GpuMemory, MutAnyOrigin](unsafe_from_address=self.gpu)
                _ = p.take_pointee()
                p.free()
                self.gpu = 0

    def __del__(deinit self):
        self._free_gpu()


struct DeviceTable(ImplicitlyCopyable):
    """The columns of a `SharedTable` on the GPU, uploaded once and shared by every session that holds a copy."""
    var _s: ArcPointer[_DeviceState]

    def __init__(out self, table: SharedTable) raises:
        """Uploads `table` to the default GPU (device 0)."""
        comptime if HAS_GPU:
            if not gpu_available():
                raise Error("DeviceTable: no GPU found")
            self = Self(table, DeviceContext())
        else:
            raise Error(NO_GPU_MESSAGE)

    def __init__(out self, table: SharedTable, ctx: DeviceContext) raises:
        """Uploads `table` on the app's `DeviceContext`: the app's buffers and these share one device."""
        comptime if HAS_GPU:
            self._s = ArcPointer(_DeviceState(table, table.rows()))
            var mem = unsafe_alloc[_GpuMemory](1)
            mem.unsafe_write(_GpuMemory(ctx.copy()))
            self._s[].gpu = Int(mem)
            ref s = self._s[]
            var n = table.rows()
            for c in range(table.columns()):
                var buf = ctx.enqueue_create_buffer[DType.int64](max(n, 1))
                if n > 0:                                # an empty column has no host memory to copy from
                    ctx.enqueue_copy(dst_buf=buf, src_ptr=table.values(c).unsafe_ptr())
                s.data_addr.append(Int(buf.unsafe_ptr()))
                mem[].data.append(buf^)
                s.bytes += 8 * max(n, 1)
                var nbytes = len(table._table[].cols[c].validity)
                if nbytes > 0:
                    var vb = ctx.enqueue_create_buffer[DType.uint8](nbytes)
                    ctx.enqueue_copy(dst_buf=vb, src_ptr=table._table[].cols[c].validity.unsafe_ptr())
                    s.valid_addr.append(Int(vb.unsafe_ptr()))
                    mem[].valid.append(vb^)
                    s.bytes += nbytes
                else:
                    s.valid_addr.append(0)
            ctx.synchronize()                            # the host columns may be freed or changed after this returns
        else:
            raise Error(NO_GPU_MESSAGE)

    def context(self) raises -> DeviceContext:
        comptime if HAS_GPU:
            if self._s[].gpu == 0:
                raise Error("the DeviceTable was released")
            return UnsafePointer[_GpuMemory, MutAnyOrigin](unsafe_from_address=self._s[].gpu)[].ctx.copy()
        else:
            raise Error(NO_GPU_MESSAGE)

    def rows(self) -> Int:
        return self._s[].rows

    def bytes(self) -> Int:
        """Device memory held (0 after `release`)."""
        return self._s[].bytes if self._s[].live else 0

    def is_live(self) -> Bool:
        return self._s[].live

    def mirrors(self, table: SharedTable) -> Bool:
        """Was this made from `table` (the same shared columns, not merely an equal-looking table)? The device table
        holds the table's handle, so the columns it mirrors cannot be freed and their address reused by another."""
        return Int(self._s[].table._table[].cols.unsafe_ptr()) == Int(table._table[].cols.unsafe_ptr())

    def data_address(self, c: Int) -> Int:
        return self._s[].data_addr[c]

    def validity_address(self, c: Int) -> Int:
        return self._s[].valid_addr[c]

    def release(self):
        """Frees the device memory for every copy of this handle (the document closed). Call it when no query on this
        table is running: the app queries from one thread, so there is no lock, and a launch still in flight would
        read freed memory."""
        ref s = self._s[]
        if not s.live:
            return
        s._free_gpu()
        s.data_addr = List[Int]()
        s.valid_addr = List[Int]()
        s.bytes = 0
        s.live = False


def _new_context() raises -> Int:
    """A DeviceContext on the default GPU, on the heap, by address (0 on a build without GPU code): a streamed scan makes
    one and uploads each decoded row group on it, so no GPU symbol is named outside `comptime if HAS_GPU`."""
    comptime if HAS_GPU:
        if not gpu_available():
            return 0
        var mem = unsafe_alloc[DeviceContext](1)
        mem.unsafe_write(DeviceContext())
        return Int(mem)
    else:
        return 0


def _free_context(addr: Int):
    comptime if HAS_GPU:
        if addr != 0:
            var p = UnsafePointer[DeviceContext, MutAnyOrigin](unsafe_from_address=addr)
            _ = p.take_pointee()
            p.free()
