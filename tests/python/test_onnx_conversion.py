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
    ConversionMetadata,
    ExecutionProvider,
    InferenceSession,
    SessionConfig,
    TensorSpec,
    build_conversion_contract,
    check_numerical_equivalence,
    check_onnx_graph,
    clear_cache,
    export_torch_model_to_onnx,
    get_default_cache_dir,
    make_dynamic_axes,
    resolve_cache_path,
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

    with pytest.raises(ValueError, match="sample_inputs is required"):
        export_torch_model_to_onnx(identity_model, None, "unused.onnx")


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_build_conversion_contract_metadata(tiny_mlp):
    """Validates that build_conversion_contract infers shapes, dtypes, and dynamic axes."""
    model, x = tiny_mlp
    contract = build_conversion_contract(
        model,
        x,
        dynamic_batch=True,
        input_names=["features"],
        output_names=["logits"],
    )

    assert isinstance(contract, ConversionMetadata)
    assert contract.model_name == "TinyMLP"
    assert contract.dynamic_batch is True
    assert len(contract.inputs) == 1
    assert contract.inputs[0] == TensorSpec(
        name="features", shape=(-1, 4), dtype="float32"
    )
    assert len(contract.outputs) == 1
    assert contract.outputs[0] == TensorSpec(
        name="logits", shape=(-1, 3), dtype="float32"
    )
    assert contract.dynamic_axes == {
        "features": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
@pytest.mark.parametrize(
    "rank,shape", [(1, (4,)), (2, (2, 4)), (3, (2, 3, 4)), (4, (2, 3, 8, 8))]
)
def test_build_conversion_contract_supports_ranks_1d_to_4d(identity_model, rank, shape):
    """Validates that 1D, 2D, 3D, and 4D tensor inputs are supported in conversion contract."""
    x = torch.randn(shape, dtype=torch.float32)
    contract = build_conversion_contract(identity_model, x, dynamic_batch=True)

    expected_shape = (-1, *shape[1:])
    assert contract.inputs[0].shape == expected_shape
    assert contract.inputs[0].dtype == "float32"
    assert contract.outputs[0].shape == expected_shape


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_build_conversion_contract_dict_outputs():
    """Validates dictionary outputs and automatic output name resolution."""

    class DictModule(torch.nn.Module):
        def forward(self, x):
            return {"logits": x * 2.0, "probabilities": torch.softmax(x, dim=-1)}

    model = DictModule().eval()
    x = torch.randn(2, 4, dtype=torch.float32)
    contract = build_conversion_contract(model, x, dynamic_batch=True)

    assert [spec.name for spec in contract.outputs] == ["logits", "probabilities"]
    assert contract.outputs[0].shape == (-1, 4)
    assert contract.outputs[1].shape == (-1, 4)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_conversion_contract_rejects_non_module():
    """Rejects objects that are not instances of torch.nn.Module."""
    with pytest.raises(TypeError, match="model must be an instance of torch.nn.Module"):
        build_conversion_contract(object(), torch.randn(2, 4))


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_conversion_contract_rejects_unsupported_dtype(identity_model):
    """Rejects tensors with dtypes outside the supported compatibility matrix."""
    x_complex = torch.randn(2, 4, dtype=torch.complex64)
    with pytest.raises(TypeError, match="Unsupported dtype 'complex64'"):
        build_conversion_contract(identity_model, x_complex)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_conversion_contract_rejects_unsupported_rank(identity_model):
    """Rejects 0D scalars and 5D+ tensors with actionable error messages."""
    x_scalar = torch.tensor(3.14, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported rank 0"):
        build_conversion_contract(identity_model, x_scalar)

    x_5d = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported rank 5"):
        build_conversion_contract(identity_model, x_5d)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_conversion_contract_rejects_mismatched_names(identity_model, tiny_mlp):
    """Rejects input and output names whose lengths do not match tensor counts."""
    model, x = tiny_mlp
    with pytest.raises(
        ValueError, match="input_names must match the number of input tensors"
    ):
        build_conversion_contract(model, x, input_names=["in1", "in2"])

    with pytest.raises(
        ValueError, match="output_names must match the number of model outputs"
    ):
        build_conversion_contract(model, x, output_names=["out1", "out2"])


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_torch_model_to_onnx_cnn_4d(tmp_path: Path):
    """Exports a 4D Conv2d CNN model and verifies round-trip inference in Pyreth."""

    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = TinyCNN().eval()
    x = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    out_path = tmp_path / "tiny_cnn.onnx"

    exported = export_torch_model_to_onnx(
        model,
        x,
        out_path,
        dynamic_batch=True,
        validate=True,
    )
    assert exported.exists()
    session = InferenceSession(str(exported))
    assert session.model_info.has_dynamic_batch
    assert session.model_info.inputs[0].shape == (-1, 3, 16, 16)
    assert session.model_info.outputs[0].shape == (-1, 8, 16, 16)

    test_input = np.random.randn(3, 3, 16, 16).astype(np.float32)
    actual = session.run_numpy([test_input])[0]
    expected = model(torch.from_numpy(test_input)).detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_torch_model_to_onnx_dict_outputs(tmp_path: Path):
    """Exports a model returning a dictionary and verifies all outputs match."""

    class MultiHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reg = torch.nn.Linear(4, 1)
            self.cls = torch.nn.Linear(4, 5)

        def forward(self, x):
            return {"regression": self.reg(x), "classification": self.cls(x)}

    model = MultiHead().eval()
    x = torch.randn(2, 4, dtype=torch.float32)
    out_path = tmp_path / "multi_head.onnx"

    exported = export_torch_model_to_onnx(
        model,
        x,
        out_path,
        dynamic_batch=True,
        validate=True,
    )
    session = InferenceSession(str(exported))
    assert [spec.name for spec in session.model_info.outputs] == [
        "regression",
        "classification",
    ]
    assert session.model_info.outputs[0].shape == (-1, 1)
    assert session.model_info.outputs[1].shape == (-1, 5)

    actual = session.run_numpy([x.numpy()])
    expected = model(x)
    np.testing.assert_allclose(
        actual[0], expected["regression"].detach().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        actual[1], expected["classification"].detach().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_with_custom_positional_input_names(
    tmp_path: Path, multi_input_multi_output_model
):
    """Exports positional tuple inputs with custom input_names without validation failure."""
    left = torch.randn(2, 4, dtype=torch.float32)
    right = torch.randn(2, 4, dtype=torch.float32)
    model = multi_input_multi_output_model
    out_path = tmp_path / "custom_pos_names.onnx"

    exported = export_torch_model_to_onnx(
        model,
        (left, right),
        out_path,
        input_names=["op_a", "op_b"],
        output_names=["res_sum", "res_prod"],
        dynamic_batch=True,
        validate=True,
    )
    session = InferenceSession(str(exported))
    assert [info.name for info in session.model_info.inputs] == ["op_a", "op_b"]
    assert [info.name for info in session.model_info.outputs] == ["res_sum", "res_prod"]


def test_make_dynamic_axes_builder():
    """Validates dynamic axes construction with custom batch dimension and extra axes."""
    axes = make_dynamic_axes(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        batch_axis=0,
        batch_dim_name="batch_size",
        additional_axes={
            "input_ids": {1: "sequence_length"},
            "attention_mask": {1: "sequence_length"},
        },
    )
    assert axes["input_ids"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["attention_mask"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["logits"] == {0: "batch_size"}


def test_cache_management_and_path_resolution(tmp_path: Path):
    """Verifies default cache directory, sanitized path resolution, and cache clearing."""
    default_dir = get_default_cache_dir()
    assert isinstance(default_dir, Path)
    assert default_dir.exists()

    hf_path = resolve_cache_path(
        "google/mobilenet_v2-1.0", cache_dir=tmp_path, prefix="image_cls__"
    )
    assert hf_path == tmp_path / "image_cls__google__mobilenet_v2_1.0.onnx"

    # Create dummy onnx cache files and clear
    (tmp_path / "dummy1.onnx").write_bytes(b"dummy")
    (tmp_path / "dummy2.onnx").write_bytes(b"dummy")
    (tmp_path / "other.txt").write_text("keep")

    removed = clear_cache(tmp_path)
    assert removed == 2
    assert not (tmp_path / "dummy1.onnx").exists()
    assert not (tmp_path / "dummy2.onnx").exists()
    assert (tmp_path / "other.txt").exists()


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_check_onnx_graph_integrity(tmp_path: Path, tiny_mlp):
    """Validates check_onnx_graph helper returns valid schema and detects errors."""
    model, x = tiny_mlp
    out_path = tmp_path / "graph_check.onnx"
    export_torch_model_to_onnx(model, x, out_path, validate=False)

    info = check_onnx_graph(out_path)
    assert info["model_path"] == str(out_path)
    assert info["opset_version"] == 17
    assert len(info["input_names"]) == 1
    assert len(info["output_names"]) == 1

    with pytest.raises(FileNotFoundError):
        check_onnx_graph(tmp_path / "non_existent.onnx")

    corrupt_file = tmp_path / "corrupt.onnx"
    corrupt_file.write_bytes(b"invalid protobuf contents")
    with pytest.raises(ValueError, match="Failed to load ONNX model"):
        check_onnx_graph(corrupt_file)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_check_numerical_equivalence_metrics(tmp_path: Path, tiny_mlp):
    """Checks detailed numerical metrics and failure when outputs diverge."""
    model, x = tiny_mlp
    out_path = tmp_path / "numerics.onnx"
    export_torch_model_to_onnx(model, x, out_path, validate=False)

    metrics = check_numerical_equivalence(model, out_path, x, rtol=1e-5, atol=1e-5)
    assert metrics["matches"] is True
    assert metrics["max_abs_diff"] < 1e-4
    assert metrics["max_rel_diff"] < 1e-4
    assert "mean_abs_diff" in metrics

    # Failure with diverging reference model
    diff_model = torch.nn.Linear(4, 3).eval()
    diff_model.bias.data.fill_(100.0)
    with pytest.raises(
        ValueError, match="does not match exported ONNX model within tolerance"
    ):
        check_numerical_equivalence(diff_model, out_path, x, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(torch is None or onnx is None, reason="torch/onnx are required")
def test_export_preserves_training_state_and_seed(tmp_path: Path, tiny_mlp):
    """Verifies that export restores model.training state and respects seed."""
    model, x = tiny_mlp
    model.train()  # Explicitly set training mode
    out_path = tmp_path / "deterministic.onnx"

    export_torch_model_to_onnx(model, x, out_path, seed=42, validate=True)
    assert model.training is True  # Preserved after export
    assert out_path.exists()
    assert not out_path.with_suffix(".tmp.onnx").exists()  # Cleaned up
