"""Sample 01: Zero-Copy Tensors, 64B SIMD Alignment, ArenaAllocator & MemoryPool.

Demonstrates:
  1. Creating 64-byte SIMD-aligned `pyreth.Tensor` objects and inspecting metadata
  2. Bidirectional zero-copy interoperability with `numpy.ndarray`
  3. Zero-copy reshaping (`reshape`) and strided slicing (`slice`, `to_contiguous`)
  4. Fast bump allocation with `ArenaAllocator`, pin counting, and O(1) `reset()`
  5. Reusable arena leases with `MemoryPool`

Usage:
    uv run python samples/01_tensor_and_memory.py
"""

import gc

import numpy as np
from pyreth import (
    ArenaAllocator,
    DataType,
    MemoryPool,
    Shape,
    Tensor,
)


def demo_zero_copy_numpy_interop() -> None:
    print("=== 1. Zero-Copy NumPy <-> pyreth.Tensor Interoperability ===")

    # 1a. Wrap an existing NumPy array without copying memory (StorageType.EXTERNAL)
    np_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = Tensor.from_numpy(np_arr)

    np_ptr = np_arr.__array_interface__["data"][0]
    print(f"Tensor: {tensor}")
    print(f"  StorageType : {tensor.storage_type}")
    print(f"  NumPy ptr   : 0x{np_ptr:x}")
    print(
        f"  Tensor ptr  : 0x{tensor.data_ptr:x} (Zero-copy match: {np_ptr == tensor.data_ptr})"
    )

    # Mutating the Tensor in-place immediately updates the underlying NumPy array
    tensor.fill(5.0)
    assert np.all(np_arr == 5.0)
    print(f"  After tensor.fill(5.0), np_arr[0]: {np_arr[0]}")

    # 1b. Allocate a 64-byte SIMD-aligned Tensor in Pyreth and export to NumPy
    aligned_tensor = Tensor.zeros(Shape([4, 16]), DataType.Float32)
    np_view = aligned_tensor.numpy()  # or np.asarray(aligned_tensor)
    print(
        f"\nPyreth-allocated Tensor: shape={aligned_tensor.shape}, dtype={aligned_tensor.dtype}"
    )
    print(f"  StorageType       : {aligned_tensor.storage_type}")
    print(
        f"  64-byte aligned   : {aligned_tensor.is_aligned(64)} (ptr % 64 == {np_view.ctypes.data % 64})"
    )
    print(
        f"  Zero-copy view    : {np_view.__array_interface__['data'][0] == aligned_tensor.data_ptr}"
    )


def demo_reshape_and_strided_slicing() -> None:
    print("\n=== 2. Zero-Copy Reshape & Strided Slicing ===")

    base_np = np.arange(24, dtype=np.float32).reshape(4, 6)
    base_tensor = Tensor.from_numpy(base_np)

    # Zero-copy reshape with automatic dimension inference (-1)
    reshaped = base_tensor.reshape((2, 3, -1))
    print(
        f"Reshaped (2, 3, -1) -> shape={reshaped.shape}, shares ptr={reshaped.data_ptr == base_tensor.data_ptr}"
    )

    # Zero-copy strided slice along dimension 1: start=1, end=6, step=2
    sliced = base_tensor.slice(dim=1, start=1, end=6, step=2)
    print(
        f"Sliced dim=1 [1:6:2] -> shape={sliced.shape}, strides={sliced.strides}, "
        f"is_contiguous={sliced.is_contiguous}"
    )

    # Materialize a contiguous copy when required by downstream kernels
    contiguous = sliced.to_contiguous()
    print(
        f"Contiguous copy      -> shape={contiguous.shape}, strides={contiguous.strides}, "
        f"is_contiguous={contiguous.is_contiguous}"
    )
    np.testing.assert_array_equal(contiguous.numpy(), base_np[:, 1:6:2])


def demo_arena_allocator_and_pin_safety() -> None:
    print("\n=== 3. ArenaAllocator Bump Allocation & Pin-Count Safety ===")

    arena = ArenaAllocator(block_size=64 * 1024)
    print(
        f"Initial arena: bytes_allocated={arena.bytes_allocated}, is_pinned={arena.is_pinned}"
    )

    # Allocate two tensors inside the same ArenaAllocator
    t1 = Tensor.zeros((16, 16), DataType.Float32, arena=arena)
    t2 = Tensor.create((8, 8), DataType.Float32, arena=arena)
    t2.fill(3.14)

    print(
        f"After 2 allocations: bytes_allocated={arena.bytes_allocated}, "
        f"pin_count={arena.pin_count}, can_reset={arena.can_reset()}"
    )

    # Export a zero-copy NumPy view from t2 and delete the Python Tensor handles
    view2 = t2.numpy()
    del t1, t2
    gc.collect()

    # The arena stays pinned because `view2` still references the arena's memory
    print(
        f"After deleting Tensors (NumPy view still alive): "
        f"pin_count={arena.pin_count}, can_reset={arena.can_reset()}"
    )
    assert np.allclose(view2, 3.14)

    # Releasing the NumPy view unpins the arena so it can be reset in O(1)
    del view2
    gc.collect()
    print(
        f"After deleting NumPy view: pin_count={arena.pin_count}, can_reset={arena.can_reset()}"
    )

    arena.reset()
    print(f"After arena.reset(): bytes_allocated={arena.bytes_allocated}")


def demo_memory_pool() -> None:
    print("\n=== 4. Thread-Safe MemoryPool Lease Recycling ===")

    pool = MemoryPool(initial_count=4, block_size=64 * 1024)
    print(f"MemoryPool initial available arenas: {pool.available_count}")

    # Allocate a pooled tensor (leases an arena from the pool)
    pooled_tensor = Tensor.zeros((32, 32), DataType.Float32, pool=pool)
    pooled_view = pooled_tensor.numpy()
    pooled_view[0, 0] = 42.0
    print(
        f"During active Tensor/NumPy view: storage_type={pooled_tensor.storage_type}, "
        f"available_count={pool.available_count}"
    )

    # Once both the Tensor and its NumPy views are collected, the arena automatically
    # resets in O(1) and returns to the MemoryPool.
    del pooled_tensor, pooled_view
    gc.collect()
    print(f"After views released: available_count={pool.available_count}")


def main() -> None:
    demo_zero_copy_numpy_interop()
    demo_reshape_and_strided_slicing()
    demo_arena_allocator_and_pin_safety()
    demo_memory_pool()


if __name__ == "__main__":
    main()
