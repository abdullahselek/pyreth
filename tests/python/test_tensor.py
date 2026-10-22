import gc
import weakref

import numpy as np
import pytest
from pyreth import ArenaAllocator, DataType, MemoryPool, Shape, StorageType, Tensor

# ---------------------------------------------------------------------------
# 1. Bidirectional Zero-Copy Mapping & Pointer Identity
# ---------------------------------------------------------------------------


def test_numpy_to_tensor_zero_copy():
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    expected_ptr = arr.__array_interface__["data"][0]

    tensor = Tensor.from_numpy(arr)
    assert tensor.is_valid
    assert tensor.is_contiguous
    assert tensor.storage_type == StorageType.EXTERNAL
    assert tensor.dtype == DataType.Float32
    assert tensor.shape == (4, 6)
    assert tensor.strides == (6, 1)
    assert tensor.byte_strides == (24, 4)
    assert tensor.ndim == 2
    assert tensor.rank == 2
    assert tensor.size == 24
    assert tensor.nbytes == 96
    assert tensor.itemsize == 4
    assert tensor.data_ptr == expected_ptr

    # Constructor overload Tensor(arr) also performs zero-copy wrapping
    tensor2 = Tensor(arr)
    assert tensor2.data_ptr == expected_ptr
    assert tensor2.storage_type == StorageType.EXTERNAL


def test_tensor_to_numpy_zero_copy_and_simd_alignment():
    tensor = Tensor.zeros((8, 16), DataType.Float32)
    assert tensor.is_valid
    assert tensor.is_contiguous
    assert tensor.storage_type == StorageType.OWNED
    assert tensor.is_aligned(64)

    arr = tensor.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 16)
    assert arr.dtype == np.float32
    assert arr.__array_interface__["data"][0] == tensor.data_ptr
    assert arr.ctypes.data % 64 == 0

    # np.asarray(tensor) uses __array__ protocol with zero copy
    arr_via_asarray = np.asarray(tensor)
    assert arr_via_asarray.__array_interface__["data"][0] == tensor.data_ptr


# ---------------------------------------------------------------------------
# 2. In-Place Mutation (NumPy <-> Tensor)
# ---------------------------------------------------------------------------


def test_in_place_mutation_numpy_to_tensor_and_back():
    arr = np.zeros((3, 4), dtype=np.float32)
    tensor = Tensor.from_numpy(arr)
    view = tensor.to_numpy()

    # Mutate original NumPy array -> visible in Tensor view
    arr[1, 2] = 42.5
    assert view[1, 2] == pytest.approx(42.5)

    # Mutate via Tensor.fill() -> visible in original NumPy array
    tensor.fill(7.25)
    assert np.all(arr == pytest.approx(7.25))

    # Mutate via Tensor.zero() -> visible in original NumPy array
    tensor.zero()
    assert np.all(arr == 0.0)


def test_in_place_mutation_tensor_to_numpy():
    tensor = Tensor.create((4, 4), DataType.Int32)
    tensor.zero()

    view1 = tensor.numpy()
    view2 = tensor.numpy()

    view1[0, 0] = 100
    view1[3, 3] = -999

    assert view2[0, 0] == 100
    assert view2[3, 3] == -999

    tensor.fill(55)
    assert np.all(view1 == 55)
    assert np.all(view2 == 55)


# ---------------------------------------------------------------------------
# 3. DataType Validation & Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("np_dtype", "pyreth_dtype"),
    [
        (np.float32, DataType.Float32),
        (np.float64, DataType.Float64),
        (np.float16, DataType.Float16),
        (np.int8, DataType.Int8),
        (np.uint8, DataType.UInt8),
        (np.int16, DataType.Int16),
        (np.uint16, DataType.UInt16),
        (np.int32, DataType.Int32),
        (np.uint32, DataType.UInt32),
        (np.int64, DataType.Int64),
        (np.uint64, DataType.UInt64),
        (np.bool_, DataType.Bool),
    ],
)
def test_all_supported_dtypes_roundtrip_and_in_place_mutation(np_dtype, pyreth_dtype):
    # 1. NumPy -> Tensor -> NumPy
    arr = np.zeros((2, 5), dtype=np_dtype)
    tensor = Tensor.from_numpy(arr)
    assert tensor.dtype == pyreth_dtype
    assert tensor.data_ptr == arr.__array_interface__["data"][0]

    roundtrip = tensor.numpy()
    assert roundtrip.dtype == np.dtype(np_dtype)
    assert roundtrip.__array_interface__["data"][0] == tensor.data_ptr

    # Mutate in-place
    val = True if np_dtype == np.bool_ else 7
    arr[1, 3] = val
    assert roundtrip[1, 3] == val

    # 2. Tensor.create -> NumPy
    owned = Tensor.zeros((3, 3), pyreth_dtype)
    assert owned.dtype == pyreth_dtype
    owned_np = owned.numpy()
    assert owned_np.dtype == np.dtype(np_dtype)
    assert owned_np.__array_interface__["data"][0] == owned.data_ptr


def test_dtype_string_and_numpy_dtype_arguments():
    t1 = Tensor.zeros((2, 2), "float32")
    assert t1.dtype == DataType.Float32

    t2 = Tensor.zeros((2, 2), np.float64)
    assert t2.dtype == DataType.Float64

    t3 = Tensor.zeros((2, 2), np.dtype("int64"))
    assert t3.dtype == DataType.Int64


def test_bfloat16_tensor_creation_and_numpy_export_error():
    t = Tensor.zeros((2, 4), DataType.BFloat16)
    assert t.dtype == DataType.BFloat16
    assert t.itemsize == 2
    assert t.nbytes == 16

    with pytest.raises(TypeError, match="BFloat16"):
        _ = t.numpy()


@pytest.mark.parametrize(
    "bad_arr",
    [
        np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
        np.array(["hello", "world"]),
        np.array([{"a": 1}, {"b": 2}], dtype=object),
    ],
)
def test_unsupported_numpy_dtypes_rejected(bad_arr):
    with pytest.raises(TypeError):
        Tensor.from_numpy(bad_arr)


def test_readonly_numpy_array_rejected():
    arr = np.ones((4, 4), dtype=np.float32)
    arr.flags.writeable = False

    with pytest.raises(ValueError, match="read-only"):
        Tensor.from_numpy(arr)


def test_negative_stride_numpy_array_rejected():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1]
    assert arr.strides[0] < 0

    with pytest.raises(ValueError, match="Negative strides"):
        Tensor.from_numpy(arr)


# ---------------------------------------------------------------------------
# 4. Reference Counting & Lifetime Management
# ---------------------------------------------------------------------------


def test_numpy_lifetime_retained_by_tensor():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr_ref = weakref.ref(arr)
    expected_ptr = arr.__array_interface__["data"][0]

    tensor = Tensor.from_numpy(arr)
    assert tensor.data_ptr == expected_ptr

    # Delete Python reference to the NumPy array; Tensor must keep it alive
    del arr
    gc.collect()
    assert arr_ref() is not None

    view = tensor.numpy()
    assert np.array_equal(view, np.arange(16, dtype=np.float32).reshape(4, 4))

    # Delete Tensor and view; underlying NumPy array can now be collected
    del view
    del tensor
    gc.collect()
    assert arr_ref() is None


def test_tensor_storage_and_arena_pinning_across_numpy_views():
    arena = ArenaAllocator(64 * 1024)
    assert not arena.is_pinned
    assert arena.pin_count == 0

    tensor = Tensor.zeros((8, 8), DataType.Float32, arena=arena)
    assert tensor.is_arena_backed
    assert tensor.storage_type == StorageType.ARENA
    assert arena.is_pinned
    assert arena.pin_count == 1
    assert tensor.use_count == 1

    view = tensor.numpy()
    view[:] = 3.14159
    # Exporting to NumPy attaches a capsule holding shared_ptr<TensorStorage>
    assert tensor.use_count == 2

    # Destroy the pyreth.Tensor handle while keeping the NumPy array alive
    del tensor
    gc.collect()

    # The ArenaAllocator must remain pinned because `view` still references the memory
    assert arena.is_pinned
    assert arena.pin_count == 1
    assert not arena.can_reset()
    assert np.allclose(view, 3.14159)

    # Once the NumPy view is deleted and collected, the arena is automatically unpinned
    del view
    gc.collect()
    assert not arena.is_pinned
    assert arena.pin_count == 0
    assert arena.can_reset()
    arena.reset()
    assert arena.bytes_allocated == 0


def test_memory_pool_lease_retained_by_numpy_view():
    pool = MemoryPool(initial_count=2, block_size=64 * 1024)
    assert pool.available_count == 2

    tensor = Tensor.zeros((4, 4), DataType.Float32, pool=pool)
    assert tensor.storage_type == StorageType.POOLED
    assert pool.available_count == 1

    view = tensor.numpy()
    view[0, 0] = 99.0

    # Delete Tensor; `view` still holds the PooledArenaStorage lease
    del tensor
    gc.collect()
    assert pool.available_count == 1
    assert view[0, 0] == pytest.approx(99.0)

    # Delete `view`; lease automatically resets and returns to MemoryPool
    del view
    gc.collect()
    assert pool.available_count == 2


# ---------------------------------------------------------------------------
# 5. Reshape, Slicing, Strided Views & 0-D Scalars
# ---------------------------------------------------------------------------


def test_zero_copy_reshape_and_strided_slicing():
    base_np = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = Tensor.from_numpy(base_np)

    # Zero-copy reshape with inferred dimension (-1)
    reshaped = tensor.reshape((2, 3, -1))
    assert reshaped.shape == (2, 3, 4)
    assert reshaped.data_ptr == tensor.data_ptr
    assert np.array_equal(reshaped.numpy(), base_np.reshape(2, 3, 4))

    # Zero-copy slice along axis 1 (non-contiguous strided view)
    sliced = tensor.slice(1, 1, 5, 2)
    assert sliced.shape == (4, 2)
    assert not sliced.is_contiguous
    sliced_np = sliced.numpy()
    assert np.array_equal(sliced_np, base_np[:, 1:5:2])

    # Mutate through strided NumPy view -> mutates original base_np in-place
    sliced_np[0, 0] = -777.0
    assert base_np[0, 1] == -777.0

    # Convert non-contiguous slice to contiguous Tensor
    contig = sliced.to_contiguous()
    assert contig.is_contiguous
    assert contig.data_ptr != sliced.data_ptr
    assert np.array_equal(contig.numpy(), base_np[:, 1:5:2])


def test_scalar_0d_tensor_roundtrip():
    scalar_np = np.array(12.5, dtype=np.float32)
    t = Tensor.from_numpy(scalar_np)
    assert t.ndim == 0
    assert t.shape == ()
    assert t.size == 1
    assert t.data_ptr == scalar_np.__array_interface__["data"][0]

    view = t.numpy()
    assert view.ndim == 0
    assert view.shape == ()
    assert float(view) == pytest.approx(12.5)

    view[()] = 99.25
    assert float(scalar_np) == pytest.approx(99.25)

    with pytest.raises(TypeError):
        _ = len(t)


def test_shape_class_bindings():
    s = Shape([2, 3, 4])
    assert len(s) == 3
    assert s.rank == 3
    assert s.ndim == 3
    assert s.dims == (2, 3, 4)
    assert s.num_elements() == 24
    assert not s.is_scalar()
    assert s[0] == 2
    assert s[-1] == 4
    assert s == Shape([2, 3, 4])
