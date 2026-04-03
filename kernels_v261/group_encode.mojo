"""group_encode: assign dense integer group IDs to a flat int32 key array.

Replaces the PyArrow dictionary_encode → np.unique CPU pipeline with a
single GPU pass.  Uses open-addressing linear-probing hash tables stored in
global device memory.

Design
------
  hash table slots:  int32[capacity]  initialised to EMPTY (INT32_MIN)
  id table:          int32[capacity]  (-1 until a thread wins the slot)
  id counter:        int32[1]         atomic monotone counter 0 … n_groups-1

For each element i in parallel:
  1. Compute slot = murmur_hash(keys[i]) % capacity  (linear probe on collision)
  2. Atomically compare-exchange slot to claim ownership of the key
  3. On first claim: atomically increment id_counter → assign a dense ID
  4. Write group_ids[i] = id stored in id_table[slot]

Output
------
  group_ids [N]    int32   dense IDs in [0, n_groups)
  n_groups_out [1] int32   total distinct groups found

Constraints
-----------
- keys must be int32 (encode string/categorical keys to int32 before calling)
- capacity must be a power-of-two >= 2 * max_expected_distinct_values
  (passed as a [1] int32 tensor; computed by Python caller)
- INT32_MIN (-2147483648) is used as the empty-slot sentinel; that value
  must not appear as a valid key.  Python caller should add 1 to any
  key range that includes INT32_MIN.

CPU fallback
------------
The CPU version uses the same open-addressing algorithm sequentially so
results match exactly.  For CPU, the existing _build_group_ids Python path
is faster, but the kernel is provided for portability.
"""

import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime EMPTY_KEY: Int32 = -2147483648  # INT32_MIN — sentinel for unused slot


# ── Murmur-inspired 32-bit finaliser (no multiply overflow on Mojo ints) ────

fn _hash32(key: Int32) -> Int32:
    """Avalanche-quality 32-bit hash with no large multiplies."""
    var k = UInt32(Int(key))
    k ^= k >> 16
    k ^= UInt32(0x45d9f3b)
    k ^= k >> 16
    k ^= k >> 4
    k ^= UInt32(0x27d4eb2f)
    k ^= k >> 15
    return Int32(Int(k) & 0x7FFFFFFF)  # clear sign bit so result >= 0


# ── CPU: sequential open-addressing insert ──────────────────────────────────

fn _group_encode_cpu(
    group_ids: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    n_groups_out: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    keys: ManagedTensorSlice[dtype=DType.int32, rank=1],
    capacity_in: ManagedTensorSlice[dtype=DType.int32, rank=1],
    key_table: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    id_table: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
):
    var n = keys.dim_size(0)
    var cap = Int(capacity_in[0])

    # Initialise hash table
    for i in range(cap):
        key_table[i] = EMPTY_KEY
        id_table[i] = -1

    var id_counter: Int32 = 0

    for i in range(n):
        var k = keys[i]
        var h = Int(_hash32(k)) % cap

        while True:
            if key_table[h] == k:
                # Key already inserted by a previous iteration
                group_ids[i] = id_table[h]
                break
            if key_table[h] == EMPTY_KEY:
                # Claim this slot
                key_table[h] = k
                id_table[h] = id_counter
                group_ids[i] = id_counter
                id_counter += 1
                break
            # Collision: linear probe
            h = (h + 1) % cap

    n_groups_out[0] = id_counter


# ── GPU: each thread inserts its key, atomic CAS on slot ────────────────────

fn _group_encode_gpu(
    group_ids: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    n_groups_out: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    keys: ManagedTensorSlice[dtype=DType.int32, rank=1],
    capacity_in: ManagedTensorSlice[dtype=DType.int32, rank=1],
    key_table: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    id_table: ManagedTensorSlice[mut=True, dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var n = keys.dim_size(0)
    var cap = Int(capacity_in[0])

    if n == 0:
        n_groups_out[0] = 0
        return

    # Pass 1: initialise hash table slots to EMPTY_KEY / -1
    @parameter
    fn init_kernel(cap: Int):
        var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        if tid < cap:
            key_table[tid] = EMPTY_KEY
            id_table[tid] = -1

    var init_blocks = ceildiv(cap, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[init_kernel](
        cap,
        grid_dim=init_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Initialise id_counter to 0
    @parameter
    fn zero_counter(_cap: Int):
        if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
            n_groups_out[0] = 0

    ctx.get_device_context().enqueue_function_experimental[zero_counter](
        cap,
        grid_dim=1,
        block_dim=1,
    )

    # Pass 2: each thread claims its key slot and records its dense ID.
    #
    # Algorithm per thread (element index `i`):
    #   h = hash(key) % cap
    #   loop:
    #     probe = atomicCAS(key_table[h], EMPTY_KEY, key)
    #     if probe == EMPTY_KEY:
    #       # we won the empty slot — assign a new dense ID
    #       new_id = atomicAdd(n_groups_out[0], 1)
    #       id_table[h] = new_id
    #       group_ids[i] = new_id
    #       break
    #     if probe == key:
    #       # key already present — wait until id_table[h] is written
    #       while id_table[h] == -1: spin
    #       group_ids[i] = id_table[h]
    #       break
    #     h = (h + 1) % cap   # linear probe
    @parameter
    fn insert_kernel(n: Int, cap: Int):
        var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        if i >= n:
            return

        var k = keys[i]
        var h = Int(_hash32(k)) % cap

        while True:
            # Attempt to claim slot h with our key
            var probe = Int(Atomic.compare_exchange_weak(
                key_table.unsafe_ptr() + h,
                EMPTY_KEY,   # expected (empty)
                k,            # desired  (our key)
            ))

            if probe == Int(EMPTY_KEY):
                # We won the empty slot — get a fresh dense ID
                var new_id = Int(Atomic.fetch_add(n_groups_out.unsafe_ptr(), Int32(1)))
                id_table[h] = Int32(new_id)
                group_ids[i] = Int32(new_id)
                return

            if probe == Int(k):
                # Slot already holds our key — spin until the winner writes id_table
                var assigned: Int32 = id_table[h]
                while assigned == -1:
                    assigned = id_table[h]
                group_ids[i] = assigned
                return

            # Different key in this slot — linear probe
            h = (h + 1) % cap

    var num_blocks = ceildiv(n, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[insert_kernel](
        n,
        cap,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# ── Kernel registration ──────────────────────────────────────────────────────

@compiler.register("group_encode")
struct GroupEncode:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        group_ids: OutputTensor[dtype=DType.int32, rank=1],
        n_groups_out: OutputTensor[dtype=DType.int32, rank=1],
        keys: InputTensor[dtype=DType.int32, rank=1],
        capacity_in: InputTensor[dtype=DType.int32, rank=1],
        key_table: OutputTensor[dtype=DType.int32, rank=1],
        id_table: OutputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _group_encode_cpu(group_ids, n_groups_out, keys, capacity_in, key_table, id_table)
        elif target == "gpu":
            _group_encode_gpu(group_ids, n_groups_out, keys, capacity_in, key_table, id_table, ctx)
        else:
            raise Error("No known target:", target)
