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


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_torch_model_to_onnx_round_trip(tmp_path: Path):
    """Exports a model and confirms Pyreth inference matches PyTorch."""

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    model = TinyMLP().eval()
    x = torch.randn(2, 4, dtype=torch.float32)
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
def test_inference_session_from_torch_wrapper(tmp_path: Path):
    """Creates an inference session directly from a PyTorch model."""

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, features):
            return self.linear(features)

    model = TinyClassifier().eval()
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
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
def test_export_rejects_invalid_contract():
    """Rejects missing sample inputs before attempting export."""

    class BadModel(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = BadModel().eval()
    with pytest.raises(ValueError):
        export_torch_model_to_onnx(model, None, "unused.onnx")
