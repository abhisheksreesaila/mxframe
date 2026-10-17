"""Thrift compact protocol, the encoding of Parquet's footer and page headers (only what Parquet uses).

A struct is a run of fields, each a header byte (field-id delta << 4 | type, or 0 | type then the id as a zigzag
varint) and a value, closed by a 0 byte. Integers are zigzag varints, binaries a varint length and the bytes,
lists a header (size << 4 | element type, or 0xF0 | type then the size as a varint) and the elements.
"""

comptime T_STOP = 0
comptime T_TRUE = 1
comptime T_FALSE = 2
comptime T_BYTE = 3
comptime T_I16 = 4
comptime T_I32 = 5
comptime T_I64 = 6
comptime T_DOUBLE = 7
comptime T_BINARY = 8
comptime T_LIST = 9
comptime T_SET = 10
comptime T_MAP = 11
comptime T_STRUCT = 12


struct ThriftWriter(Movable):
    var out: List[UInt8]
    var last: List[Int]          # the last field id of each open struct

    def __init__(out self):
        self.out = List[UInt8]()
        self.last = [0]

    def varint(mut self, v: UInt64):
        var x = v
        while x >= 0x80:
            self.out.append(UInt8(x & 0x7F) | 0x80)
            x >>= 7
        self.out.append(UInt8(x))

    def zigzag(mut self, v: Int64):
        self.varint(UInt64((v << 1) ^ (v >> 63)))

    def field(mut self, id: Int, type: Int):
        var delta = id - self.last[len(self.last) - 1]
        if 0 < delta <= 15:
            self.out.append(UInt8((delta << 4) | type))
        else:
            self.out.append(UInt8(type))
            self.zigzag(Int64(id))
        self.last[len(self.last) - 1] = id

    def i32(mut self, id: Int, v: Int):
        self.field(id, T_I32)
        self.zigzag(Int64(v))

    def i64(mut self, id: Int, v: Int64):
        self.field(id, T_I64)
        self.zigzag(v)

    def boolean(mut self, id: Int, v: Bool):
        self.field(id, T_TRUE if v else T_FALSE)

    def binary(mut self, id: Int, s: String):
        self.field(id, T_BINARY)
        self.raw_binary(s)

    def raw_binary(mut self, s: String):
        var b = s.as_bytes()
        self.varint(UInt64(len(b)))
        for c in b:
            self.out.append(c)

    def begin_struct(mut self, id: Int):
        """A struct-valued field: its fields follow, then `end_struct`."""
        self.field(id, T_STRUCT)
        self.last.append(0)

    def begin_list_element_struct(mut self):
        """A struct as a list element (no field header)."""
        self.last.append(0)

    def end_struct(mut self):
        self.out.append(T_STOP)
        _ = self.last.pop()

    def list_header(mut self, id: Int, elem_type: Int, size: Int):
        self.field(id, T_LIST)
        if size < 15:
            self.out.append(UInt8((size << 4) | elem_type))
        else:
            self.out.append(UInt8(0xF0 | elem_type))
            self.varint(UInt64(size))

    def end(mut self):
        """Closes the top-level struct."""
        self.out.append(T_STOP)

    def take(deinit self) -> List[UInt8]:
        """The encoded bytes (the writer is used up)."""
        return self.out^


struct ThriftReader(Movable):
    """Reads compact-protocol structs from `n` bytes at `base`."""
    var base: Int
    var n: Int
    var pos: Int
    var last: List[Int]

    def __init__(out self, base: Int, n: Int):
        self.base = base
        self.n = n
        self.pos = 0
        self.last = [0]

    def byte(mut self) raises -> Int:
        if self.pos >= self.n:
            raise Error("Parquet metadata ends early (corrupt or truncated file)")
        var b = Int(UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.base)[self.pos])
        self.pos += 1
        return b

    def varint(mut self) raises -> UInt64:
        var v: UInt64 = 0
        var shift = 0
        while True:
            var b = self.byte()
            v |= UInt64(b & 0x7F) << UInt64(shift)
            if b < 0x80:
                return v
            shift += 7
            if shift > 63:
                raise Error("Parquet metadata has a bad varint")

    def zigzag(mut self) raises -> Int64:
        var u = self.varint()
        return Int64(u >> 1) ^ -Int64(u & 1)

    def field(mut self) raises -> Tuple[Int, Int]:
        """(field id, type) of the next field; type T_STOP at the end of the struct."""
        var b = self.byte()
        var type = b & 0x0F
        if type == T_STOP:
            return (0, T_STOP)
        var delta = b >> 4
        var id = self.last[len(self.last) - 1] + delta if delta != 0 else Int(self.zigzag())
        self.last[len(self.last) - 1] = id
        return (id, type)

    def binary(mut self) raises -> String:
        var n = Int(self.varint())
        if self.pos + n > self.n:
            raise Error("Parquet metadata has a string past its end")
        var s = String(unsafe_from_utf8=Span[UInt8, MutAnyOrigin](
            unsafe_ptr=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.base + self.pos), length=n))
        self.pos += n
        return s^

    def list_header(mut self) raises -> Tuple[Int, Int]:
        """(element type, size)."""
        var b = self.byte()
        var size = b >> 4
        if size == 15:
            size = Int(self.varint())
        return (b & 0x0F, size)

    def enter(mut self):
        self.last.append(0)

    def leave(mut self):
        _ = self.last.pop()

    def skip(mut self, type: Int) raises:
        """Skips a value of `type` (fields Parquet readers may ignore)."""
        if type == T_TRUE or type == T_FALSE:
            return
        if type == T_BYTE:
            _ = self.byte()
        elif type == T_I16 or type == T_I32 or type == T_I64:
            _ = self.varint()
        elif type == T_DOUBLE:
            self.pos += 8
        elif type == T_BINARY:
            var n = Int(self.varint())
            self.pos += n
        elif type == T_LIST or type == T_SET:
            var h = self.list_header()
            for _ in range(h[1]):
                if h[0] == T_TRUE or h[0] == T_FALSE:
                    _ = self.byte()          # a bool list element is one byte
                else:
                    self.skip(h[0])
        elif type == T_MAP:
            var size = Int(self.varint())
            if size > 0:
                var kv = self.byte()
                for _ in range(size):
                    self.skip(kv >> 4)
                    self.skip(kv & 0x0F)
        elif type == T_STRUCT:
            self.enter()
            while True:
                var f = self.field()
                if f[1] == T_STOP:
                    break
                self.skip(f[1])
            self.leave()
        else:
            raise Error("Parquet metadata has an unknown thrift type " + String(type))
