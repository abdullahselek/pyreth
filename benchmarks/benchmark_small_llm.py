"""Small LLM HuggingFace-to-ONNX Prefill Benchmark Suite for Pyreth.

Downloads a small open-source causal language model from HuggingFace Hub
(default: `HuggingFaceTB/SmolLM2-135M`), wraps it for prefill inference
(`use_cache=False`), exports it to ONNX with a dynamic `batch_size` axis,
verifies numerical equivalence against PyTorch, and runs Pyreth's 4-mode
serving benchmark suite across batch sizes 1, 2, 4, 8, 16, 32, and 64.

Usage:
    uv run python benchmarks/benchmark_small_llm.py
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from common import (
    BATCH_SIZES,
    DEFAULT_CACHE_DIR,
    create_default_session,
    print_benchmark_report,
    run_model_benchmark_suite,
)
from pyreth import InferenceSession, Tensor
from torch import nn
from transformers import AutoModelForCausalLM

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_SEQ_LEN = 32


class CausalLMPrefillWrapper(nn.Module):
    """Thin wrapper around a HuggingFace causal language model for prefill ONNX export.

    Calls `model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits`
    so the exported ONNX graph accepts `(input_ids, attention_mask)` (`int64`,
    `[batch_size, seq_len]`) and outputs raw vocabulary `logits` (`float32`,
    `[batch_size, seq_len, vocab_size]`).

    Attributes:
        model: Pretrained HuggingFace `AutoModelForCausalLM` instance.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the prefill wrapper with a pretrained causal language model.

        Args:
            model: Pretrained PyTorch causal LM `nn.Module`.
        """
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run prefill forward pass (`use_cache=False`) and return vocabulary logits.

        Args:
            input_ids: Token ID tensor of shape `[batch_size, seq_len]` (`int64`).
            attention_mask: Attention mask tensor of shape `[batch_size, seq_len]` (`int64`).

        Returns:
            Unnormalized token logits tensor of shape `[batch_size, seq_len, vocab_size]`
            (`float32`).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


def _sanitize_model_filename(model_id: str, seq_len: int) -> str:
    """Build a filesystem-safe cache filename for an exported ONNX causal LM model.

    Args:
        model_id: HuggingFace repository identifier (e.g., `HuggingFaceTB/SmolLM2-135M`).
        seq_len: Fixed prompt sequence length used during export.

    Returns:
        Sanitized `.onnx` filename string.
    """
    safe_name = model_id.replace("/", "__").replace("-", "_")
    return f"small_llm__{safe_name}__seq{seq_len}.onnx"


def export_small_llm_to_onnx(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seq_len: int = DEFAULT_SEQ_LEN,
    force_export: bool = False,
    verify_numerics: bool = True,
) -> tuple[Path, int, int]:
    """Download a causal LM from HuggingFace and export its prefill graph to ONNX.

    Args:
        model_id: HuggingFace causal language model identifier to download.
        cache_dir: Local directory where the exported `.onnx` file is cached.
        seq_len: Sequence length for `input_ids` and `attention_mask`.
        force_export: If True, re-export the ONNX file even if already cached.
        verify_numerics: If True, validate dynamic batch metadata and PyTorch equivalence.

    Returns:
        A tuple of `(onnx_path, seq_len, vocab_size)`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / _sanitize_model_filename(model_id, seq_len)

    if onnx_path.exists() and not force_export and not verify_numerics:
        session = create_default_session(onnx_path)
        in_info = session.get_input_info("input_ids")
        out_info = session.get_output_info("logits")
        return onnx_path, int(in_info.shape[1]), int(out_info.shape[2])

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval()

    cfg = hf_model.config
    vocab_size = int(getattr(cfg, "vocab_size", 49152))
    wrapper = CausalLMPrefillWrapper(hf_model).eval()

    if not onnx_path.exists() or force_export:
        dummy_input_ids = torch.randint(0, vocab_size, (2, seq_len), dtype=torch.int64)
        dummy_attention_mask = torch.ones((2, seq_len), dtype=torch.int64)
        tmp_path = onnx_path.with_suffix(".tmp.onnx")

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_input_ids, dummy_attention_mask),
                str(tmp_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )

        # Inline any external data into a single self-contained ONNX file
        onnx_proto = onnx.load(str(tmp_path), load_external_data=True)
        onnx.save_model(onnx_proto, str(onnx_path), save_as_external_data=False)
        tmp_path.unlink(missing_ok=True)
        data_file = Path(f"{tmp_path}.data")
        data_file.unlink(missing_ok=True)

    if verify_numerics:
        verify_onnx_equivalence(wrapper, onnx_path, seq_len, vocab_size)

    return onnx_path, seq_len, vocab_size


def verify_onnx_equivalence(
    torch_wrapper: nn.Module,
    onnx_path: Path,
    seq_len: int,
    vocab_size: int,
) -> InferenceSession:
    """Verify ONNX dynamic batch metadata and numerical equivalence against PyTorch.

    Args:
        torch_wrapper: Wrapped PyTorch causal LM prefill module in eval mode.
        onnx_path: Path to the exported `.onnx` model file.
        seq_len: Input token sequence length.
        vocab_size: Expected vocabulary dimension of output logits.

    Returns:
        Validated `pyreth.InferenceSession` instance.

    Raises:
        AssertionError: If dynamic batch metadata is missing, tensor shapes mismatch,
            or ONNX Runtime outputs diverge from PyTorch.
    """
    session = create_default_session(onnx_path)
    info = session.model_info
    if not info.has_dynamic_batch:
        raise AssertionError(f"Expected dynamic batch on exported ONNX model: {info}")

    ids_info = session.get_input_info("input_ids")
    mask_info = session.get_input_info("attention_mask")
    out_info = session.get_output_info("logits")

    if ids_info.shape != (-1, seq_len):
        raise AssertionError(
            f"Unexpected input_ids shape {ids_info.shape}, expected (-1, {seq_len})"
        )
    if mask_info.shape != (-1, seq_len):
        raise AssertionError(
            f"Unexpected attention_mask shape {mask_info.shape}, expected (-1, {seq_len})"
        )
    if out_info.shape != (-1, seq_len, vocab_size):
        raise AssertionError(
            f"Unexpected logits shape {out_info.shape}, expected (-1, {seq_len}, {vocab_size})"
        )

    rng = np.random.default_rng(123)
    sample_ids = rng.integers(0, vocab_size, size=(2, seq_len), dtype=np.int64)
    sample_mask = np.ones((2, seq_len), dtype=np.int64)

    with torch.no_grad():
        expected_logits = (
            torch_wrapper(
                torch.from_numpy(sample_ids),
                torch.from_numpy(sample_mask),
            )
            .cpu()
            .numpy()
        )

    actual_logits = session.run_numpy(
        {
            "input_ids": Tensor.from_numpy(sample_ids),
            "attention_mask": Tensor.from_numpy(sample_mask),
        }
    )[0]
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-3, atol=1e-3)
    return session


def run_small_llm_benchmarks(
    model_id: str = DEFAULT_MODEL_ID,
    seq_len: int = DEFAULT_SEQ_LEN,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_export: bool = False,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    iterations: int = 4,
    warmup: int = 1,
    dynamic_requests: int = 64,
    client_threads: int = 8,
    num_workers: int = 4,
    check_regressions: bool = True,
) -> dict[str, object]:
    """Export the HuggingFace causal LM to ONNX and run all 4 Pyreth benchmark modes.

    Args:
        model_id: HuggingFace causal LM model identifier.
        seq_len: Input token sequence length.
        cache_dir: Directory to cache exported ONNX models.
        force_export: Re-export the ONNX model even if already cached.
        batch_sizes: Tuple of batch sizes to evaluate.
        iterations: Timed iterations per batch size for sync/async benchmarks.
        warmup: Warmup iterations per batch size.
        dynamic_requests: Total concurrent requests per batch size for `DynamicBatcher`.
        client_threads: Number of concurrent Python client threads for `DynamicBatcher`.
        num_workers: Number of C++ worker threads in `ThreadPool` and `DynamicBatcher`.
        check_regressions: Whether to run automated latency/throughput assertions.

    Returns:
        Dictionary containing benchmark results, environment metadata, and model details.
    """
    onnx_path, resolved_seq_len, vocab_size = export_small_llm_to_onnx(
        model_id=model_id,
        cache_dir=cache_dir,
        seq_len=seq_len,
        force_export=force_export,
        verify_numerics=True,
    )
    session = create_default_session(onnx_path)

    rng = np.random.default_rng(42)
    cached_inputs: dict[int, dict[str, Tensor]] = {}
    for b in batch_sizes:
        ids_arr = rng.integers(
            0, vocab_size, size=(b, resolved_seq_len), dtype=np.int64
        )
        mask_arr = np.ones((b, resolved_seq_len), dtype=np.int64)
        cached_inputs[b] = {
            "input_ids": Tensor.from_numpy(ids_arr),
            "attention_mask": Tensor.from_numpy(mask_arr),
        }

    def build_input(batch_size: int) -> dict[str, Tensor]:
        if batch_size not in cached_inputs:
            ids_arr = rng.integers(
                0, vocab_size, size=(batch_size, resolved_seq_len), dtype=np.int64
            )
            mask_arr = np.ones((batch_size, resolved_seq_len), dtype=np.int64)
            cached_inputs[batch_size] = {
                "input_ids": Tensor.from_numpy(ids_arr),
                "attention_mask": Tensor.from_numpy(mask_arr),
            }
        return cached_inputs[batch_size]

    report = run_model_benchmark_suite(
        session=session,
        input_builder=build_input,
        model_id=model_id,
        task_name="small_llm",
        batch_sizes=batch_sizes,
        iterations=iterations,
        warmup=warmup,
        dynamic_requests=dynamic_requests,
        client_threads=client_threads,
        num_workers=num_workers,
        check_regressions=check_regressions,
        max_p99_ms=30000.0,
        min_scaling_factor=1.1,
    )
    report["model_details"] = {
        "onnx_path": str(onnx_path),
        "seq_len": resolved_seq_len,
        "vocab_size": vocab_size,
        "input_shapes": {
            "input_ids": ["batch_size", resolved_seq_len],
            "attention_mask": ["batch_size", resolved_seq_len],
        },
        "output_shape": ["batch_size", resolved_seq_len, vocab_size],
    }
    return report


def main() -> None:
    """Parse CLI arguments and run the Small LLM prefill benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Pyreth Small LLM HuggingFace-to-ONNX Prefill Benchmark Suite"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace causal LM model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Input sequence length for prefill benchmark (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache exported ONNX models",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export the ONNX model even if cached",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Iterations per batch size for sync/async benchmarks (default: 4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per batch size (default: 1)",
    )
    parser.add_argument(
        "--dynamic-requests",
        type=int,
        default=64,
        help="Total concurrent requests per batch size for DynamicBatcher (default: 64)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Concurrent Python client threads for DynamicBatcher (default: 8)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report",
    )
    parser.add_argument(
        "--skip-regression-check",
        action="store_true",
        help="Skip automated regression assertions",
    )
    args = parser.parse_args()

    report = run_small_llm_benchmarks(
        model_id=args.model_id,
        seq_len=args.seq_len,
        cache_dir=args.cache_dir,
        force_export=args.force_export,
        iterations=args.iterations,
        warmup=args.warmup,
        dynamic_requests=args.dynamic_requests,
        client_threads=args.threads,
        check_regressions=not args.skip_regression_check,
    )
    details = report["model_details"]  # type: ignore[index]
    print_benchmark_report(
        title="Pyreth Small LLM Prefill Benchmark Suite",
        model_desc=(
            f"{args.model_id} (Input: [batch_size, {details['seq_len']}] Int64 -> "
            f"Output: {details['output_shape']} Float32)"
        ),
        report=report,
        client_threads=args.threads,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
