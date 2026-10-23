"""Image Classification HuggingFace-to-ONNX Benchmark Suite for Pyreth.

Downloads an open-source image classification model from HuggingFace Hub
(default: `google/mobilenet_v2_1.0_224`), exports it to ONNX with a dynamic
`batch_size` axis, verifies numerical equivalence against PyTorch, and runs
Pyreth's 4-mode serving benchmark suite across batch sizes 1, 2, 4, 8, 16, 32, and 64.

Usage:
    uv run python benchmarks/benchmark_image_classification.py
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
from transformers import AutoModelForImageClassification

DEFAULT_MODEL_ID = "google/mobilenet_v2_1.0_224"


class ImageClassificationLogitsWrapper(nn.Module):
    """Thin wrapper around a HuggingFace image classifier returning raw `logits` tensor.

    Attributes:
        model: Pretrained HuggingFace `AutoModelForImageClassification` instance.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the wrapper with a pretrained image classification model.

        Args:
            model: Pretrained PyTorch image classification `nn.Module`.
        """
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return the unnormalized class logits tensor.

        Args:
            pixel_values: Input image batch tensor of shape `[batch_size, 3, H, W]`.

        Returns:
            Output class logits tensor of shape `[batch_size, num_labels]`.
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def _sanitize_model_filename(model_id: str, image_size: int) -> str:
    """Build a filesystem-safe cache filename for an exported ONNX vision model.

    Args:
        model_id: HuggingFace repository identifier (e.g., `google/mobilenet_v2_1.0_224`).
        image_size: Spatial height/width resolution in pixels.

    Returns:
        Sanitized `.onnx` filename string.
    """
    safe_name = model_id.replace("/", "__").replace("-", "_")
    return f"image_cls__{safe_name}__{image_size}x{image_size}.onnx"


def export_image_classification_to_onnx(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    image_size: int = 224,
    force_export: bool = False,
    verify_numerics: bool = True,
) -> tuple[Path, int, int]:
    """Download an image classification model from HuggingFace and export to ONNX.

    Args:
        model_id: HuggingFace model identifier to download.
        cache_dir: Local directory where the exported `.onnx` file is cached.
        image_size: Default spatial input dimension if not specified in model config.
        force_export: If True, re-export the ONNX file even if already cached.
        verify_numerics: If True, validate dynamic batch metadata and PyTorch equivalence.

    Returns:
        A tuple of `(onnx_path, resolved_image_size, num_labels)`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / _sanitize_model_filename(model_id, image_size)

    if onnx_path.exists() and not force_export and not verify_numerics:
        session = create_default_session(onnx_path)
        in_info = session.get_input_info("pixel_values")
        out_info = session.get_output_info("logits")
        return onnx_path, int(in_info.shape[2]), int(out_info.shape[1])

    hf_model = AutoModelForImageClassification.from_pretrained(model_id)
    hf_model.eval()

    cfg = hf_model.config
    resolved_size = int(getattr(cfg, "image_size", image_size))
    num_labels = int(getattr(cfg, "num_labels", 1000))
    wrapper = ImageClassificationLogitsWrapper(hf_model).eval()

    if not onnx_path.exists() or force_export:
        dummy_input = torch.randn(
            2, 3, resolved_size, resolved_size, dtype=torch.float32
        )
        tmp_path = onnx_path.with_suffix(".tmp.onnx")

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_input,),
                str(tmp_path),
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
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
        verify_onnx_equivalence(wrapper, onnx_path, resolved_size, num_labels)

    return onnx_path, resolved_size, num_labels


def verify_onnx_equivalence(
    torch_wrapper: nn.Module,
    onnx_path: Path,
    image_size: int,
    num_labels: int,
) -> InferenceSession:
    """Verify ONNX dynamic batch metadata and numerical equivalence against PyTorch.

    Args:
        torch_wrapper: Wrapped PyTorch image classification model in eval mode.
        onnx_path: Path to the exported `.onnx` model file.
        image_size: Input image height and width in pixels.
        num_labels: Expected number of output classification classes.

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

    in_info = session.get_input_info("pixel_values")
    out_info = session.get_output_info("logits")
    if in_info.shape != (-1, 3, image_size, image_size):
        raise AssertionError(
            f"Unexpected input shape {in_info.shape}, expected (-1, 3, {image_size}, {image_size})"
        )
    if out_info.shape != (-1, num_labels):
        raise AssertionError(
            f"Unexpected output shape {out_info.shape}, expected (-1, {num_labels})"
        )

    rng = np.random.default_rng(123)
    sample_np = rng.standard_normal((2, 3, image_size, image_size), dtype=np.float32)

    with torch.no_grad():
        expected_logits = torch_wrapper(torch.from_numpy(sample_np)).cpu().numpy()

    actual_logits = session.run_numpy(sample_np)[0]
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-3, atol=1e-4)
    return session


def run_image_classification_benchmarks(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_export: bool = False,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    iterations: int = 20,
    warmup: int = 5,
    dynamic_requests: int = 64,
    client_threads: int = 8,
    num_workers: int = 4,
    check_regressions: bool = True,
) -> dict[str, object]:
    """Export the HuggingFace image classifier to ONNX and run all 4 Pyreth benchmark modes.

    Args:
        model_id: HuggingFace image classification model identifier.
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
    onnx_path, image_size, num_labels = export_image_classification_to_onnx(
        model_id=model_id,
        cache_dir=cache_dir,
        force_export=force_export,
        verify_numerics=True,
    )
    session = create_default_session(onnx_path)

    rng = np.random.default_rng(42)
    cached_inputs: dict[int, Tensor] = {}
    for b in batch_sizes:
        arr = rng.standard_normal((b, 3, image_size, image_size), dtype=np.float32)
        cached_inputs[b] = Tensor.from_numpy(arr)

    def build_input(batch_size: int) -> Tensor:
        if batch_size not in cached_inputs:
            arr = rng.standard_normal(
                (batch_size, 3, image_size, image_size), dtype=np.float32
            )
            cached_inputs[batch_size] = Tensor.from_numpy(arr)
        return cached_inputs[batch_size]

    report = run_model_benchmark_suite(
        session=session,
        input_builder=build_input,
        model_id=model_id,
        task_name="image_classification",
        batch_sizes=batch_sizes,
        iterations=iterations,
        warmup=warmup,
        dynamic_requests=dynamic_requests,
        client_threads=client_threads,
        num_workers=num_workers,
        check_regressions=check_regressions,
        max_p99_ms=10000.0,
        min_scaling_factor=1.1,
    )
    report["model_details"] = {
        "onnx_path": str(onnx_path),
        "input_shape": [
            "batch_size",
            3,
            image_size,
            image_size,
        ],
        "output_shape": ["batch_size", num_labels],
    }
    return report


def main() -> None:
    """Parse CLI arguments and run the Image Classification benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Pyreth Image Classification HuggingFace-to-ONNX Benchmark Suite"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
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
        default=20,
        help="Iterations per batch size for sync/async benchmarks (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per batch size (default: 5)",
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

    report = run_image_classification_benchmarks(
        model_id=args.model_id,
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
        title="Pyreth Image Classification Benchmark Suite",
        model_desc=(
            f"{args.model_id} (Input: {details['input_shape']} -> Output: {details['output_shape']}, Float32)"
        ),
        report=report,
        client_threads=args.threads,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
