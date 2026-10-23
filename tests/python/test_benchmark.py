"""Automated pytest integration tests for benchmarks/benchmark_serving.py and benchmarks/common.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from pyreth import Tensor


def _load_module(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_serving_benchmark_suite_and_regression_checks():
    bench_mod = _load_module("benchmark_serving", "benchmarks/benchmark_serving.py")
    report = bench_mod.run_all_benchmarks(
        iterations=25,
        warmup=5,
        dynamic_requests=128,
        client_threads=8,
        check_regressions=True,
    )

    assert len(report["sync_heap"]) == 7
    assert len(report["sync_arena"]) == 7
    assert len(report["async_thread_pool"]) == 7
    assert len(report["dynamic_batcher"]) == 7
    assert len(report["regression_checks"]) == 3


def test_shared_benchmark_common_single_and_multi_input():
    bench_serving = _load_module("benchmark_serving", "benchmarks/benchmark_serving.py")
    common = _load_module("benchmark_common", "benchmarks/common.py")
    sample_models = _load_module("sample_models", "samples/models.py")

    # 1. Single-input model via common.run_model_benchmark_suite
    mlp_bytes = bench_serving.build_serving_mlp_onnx_bytes(
        in_features=32, hidden_features=64, out_features=8
    )
    mlp_session = common.create_default_session(mlp_bytes)
    assert mlp_session.is_valid

    def build_mlp_input(batch_size: int) -> Tensor:
        return Tensor.from_numpy(np.ones((batch_size, 32), dtype=np.float32))

    block_size = common.compute_recommended_arena_block_size(
        mlp_session, input_builder=build_mlp_input, max_batch_size=64
    )
    assert block_size >= 1024 * 1024
    assert block_size % 65536 == 0

    mlp_report = common.run_model_benchmark_suite(
        session=mlp_session,
        input_builder=build_mlp_input,
        model_id="test/mlp-32x64x8",
        task_name="unit_test_mlp",
        batch_sizes=(1, 4, 16),
        iterations=15,
        warmup=3,
        dynamic_requests=64,
        client_threads=4,
        check_regressions=True,
    )
    assert len(mlp_report["sync_heap"]) == 3
    assert len(mlp_report["sync_arena"]) == 3
    assert len(mlp_report["async_thread_pool"]) == 3
    assert len(mlp_report["dynamic_batcher"]) == 3
    assert len(mlp_report["regression_checks"]) == 3

    # 2. Multi-input / multi-output dictionary model via common.run_model_benchmark_suite
    dual_bytes = sample_models.build_dual_io_onnx_bytes(features=16)
    dual_session = common.create_default_session(dual_bytes)

    def build_dual_inputs(batch_size: int) -> dict[str, Tensor]:
        return {
            "a": Tensor.from_numpy(np.ones((batch_size, 16), dtype=np.float32)),
            "b": Tensor.from_numpy(np.full((batch_size, 16), 2.0, dtype=np.float32)),
        }

    dual_report = common.run_model_benchmark_suite(
        session=dual_session,
        input_builder=build_dual_inputs,
        model_id="test/dual-io",
        task_name="unit_test_dual_io",
        batch_sizes=(1, 8),
        iterations=15,
        warmup=3,
        dynamic_requests=64,
        client_threads=4,
        check_regressions=True,
    )
    assert len(dual_report["sync_heap"]) == 2
    assert len(dual_report["dynamic_batcher"]) == 2
