"""Automated pytest integration test for benchmarks/benchmark_serving.py."""

import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "benchmarks" / "benchmark_serving.py"
    spec = importlib.util.spec_from_file_location("benchmark_serving", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_serving"] = module
    spec.loader.exec_module(module)
    return module


def test_serving_benchmark_suite_and_regression_checks():
    bench_mod = _load_benchmark_module()
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
