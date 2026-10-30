from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    import onnx
except ModuleNotFoundError:  # pragma: no cover
    onnx = None

from pyreth import (
    ExecutionProvider,
    InferenceSession,
    SessionConfig,
    export_torch_model_to_onnx,
)


@pytest.fixture
def tiny_mlp():
    """Return a small deterministic MLP for export tests."""
    if torch is None or onnx is None:  # pragma: no cover - guarded by test skips
        pytest.skip("torch/onnx are required")

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    model = TinyMLP().eval()
    return model, torch.randn(2, 4, dtype=torch.float32)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_torch_model_to_onnx_round_trip(tmp_path: Path, tiny_mlp):
    """Exports a model and confirms Pyreth inference matches PyTorch."""

    model, x = tiny_mlp
    out_path = tmp_path / "tiny_mlp.onnx"

    exported = export_torch_model_to_onnx(
        model,
        x,
        out_path,
        dynamic_batch=True,
        validate=True,
    )

    assert exported.exists()
    session = InferenceSession(
        str(exported),
        SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        ),
    )
    expected = model(x).detach().numpy()
    actual = session.run_numpy({"input_0": x.detach().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert session.model_info.has_dynamic_batch


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_inference_session_from_torch_wrapper(tmp_path: Path, tiny_mlp):
    """Creates an inference session directly from a PyTorch model."""

    model, sample = tiny_mlp
    session = InferenceSession.from_torch(
        model,
        sample,
        output_path=tmp_path / "tiny_classifier.onnx",
        dynamic_batch=True,
        validate=True,
    )

    expected = model(sample).detach().numpy()
    actual = session.run_numpy({"input_0": sample.detach().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_torch_model_to_onnx_uses_cache_dir(tmp_path: Path, tiny_mlp):
    """Stores cached exports beneath the configured cache directory."""

    model, x = tiny_mlp
    export_path = export_torch_model_to_onnx(
        model,
        x,
        output_path=None,
        cache_dir=tmp_path / "models",
        validate=False,
    )

    assert export_path.parent == tmp_path / "models"
    assert export_path.exists()


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_rejects_invalid_contract():
    """Rejects missing sample inputs before attempting export."""

    class BadModel(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = BadModel().eval()
    with pytest.raises(ValueError):
        export_torch_model_to_onnx(model, None, "unused.onnx")
