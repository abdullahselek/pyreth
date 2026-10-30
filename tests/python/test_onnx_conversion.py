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


@pytest.fixture
def identity_model():
    """Return a parameter-free model for conversion compatibility tests."""
    if torch is None or onnx is None:  # pragma: no cover - guarded by test skips
        pytest.skip("torch/onnx are required")
    return torch.nn.Identity().eval()


@pytest.fixture
def multi_input_multi_output_model():
    """Return a model with two inputs and two tensor outputs."""
    if torch is None or onnx is None:  # pragma: no cover - guarded by test skips
        pytest.skip("torch/onnx are required")

    class AddAndMultiply(torch.nn.Module):
        def forward(self, left, right):
            return left + right, left * right

    return AddAndMultiply().eval()


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
    assert session.model_info.inputs[0].shape == (-1, 4)
    assert session.model_info.outputs[0].shape == (-1, 3)


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
@pytest.mark.parametrize(
    ("torch_dtype", "numpy_dtype"),
    [
        (torch.float32 if torch else None, np.float32),
        (torch.float16 if torch else None, np.float16),
        (torch.int64 if torch else None, np.int64),
        (torch.int32 if torch else None, np.int32),
        (torch.bool if torch else None, np.bool_),
    ],
)
def test_export_preserves_supported_dtype_and_dynamic_batch(
    tmp_path: Path, identity_model, torch_dtype, numpy_dtype
):
    """Exports mainstream tensor dtypes with a dynamic leading batch axis."""
    if torch_dtype is torch.bool:
        sample = torch.ones((2, 4), dtype=torch_dtype)
    elif torch_dtype in (torch.int32, torch.int64):
        sample = torch.arange(8, dtype=torch_dtype).reshape(2, 4)
    else:
        sample = torch.arange(8, dtype=torch_dtype).reshape(2, 4) / 4

    exported = export_torch_model_to_onnx(
        identity_model,
        sample,
        tmp_path / f"identity_{numpy_dtype.__name__}.onnx",
        dynamic_batch=True,
        validate=True,
    )
    session = InferenceSession(str(exported))

    assert session.model_info.has_dynamic_batch
    assert session.model_info.inputs[0].shape == (-1, 4)
    assert (
        session.model_info.inputs[0].dtype.name.lower() == numpy_dtype.__name__.lower()
    )
    runtime_input = np.ones((5, 4), dtype=numpy_dtype)
    actual = session.run_numpy({"input_0": runtime_input})[0]
    np.testing.assert_array_equal(actual, runtime_input)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_supports_multiple_inputs_and_outputs(
    tmp_path: Path, multi_input_multi_output_model
):
    """Preserves named multi-input contracts and tuple model outputs."""
    left = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    right = torch.full((2, 4), 2.0, dtype=torch.float32)
    model = multi_input_multi_output_model
    exported = export_torch_model_to_onnx(
        model,
        {"left": left, "right": right},
        tmp_path / "multi_io.onnx",
        input_names=("left", "right"),
        output_names=("sum", "product"),
        dynamic_batch=True,
        validate=True,
    )
    session = InferenceSession(str(exported))

    assert session.model_info.has_dynamic_batch
    assert [info.name for info in session.model_info.inputs] == ["left", "right"]
    assert [info.name for info in session.model_info.outputs] == ["sum", "product"]
    actual = session.run_numpy(
        {"left": left.detach().numpy(), "right": right.detach().numpy()}
    )
    expected = model(left, right)
    assert len(actual) == 2
    np.testing.assert_allclose(actual[0], expected[0].detach().numpy())
    np.testing.assert_allclose(actual[1], expected[1].detach().numpy())


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_rejects_invalid_contract(identity_model):
    """Rejects missing sample inputs before attempting export."""

    with pytest.raises(ValueError):
        export_torch_model_to_onnx(identity_model, None, "unused.onnx")
