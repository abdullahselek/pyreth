# PyTorch to ONNX Conversion Contract & Compatibility Matrix

Pyreth provides a conversion and validation pipeline to export PyTorch models (`torch.nn.Module`) into ONNX graphs optimized for execution with Pyreth's zero-copy C++20 `InferenceSession` and `DynamicBatcher`.

---

## 1. Supported Compatibility Matrix

| Category | Specification | Supported Values / Behaviors |
| :--- | :--- | :--- |
| **Supported Dtypes** | Precision & types | `float32`, `float16`, `int64`, `int32`, `bool` (`SUPPORTED_DTYPES`) |
| **Tensor Ranks** | Dimensionality | **1D, 2D, 3D, and 4D** tensors (`SUPPORTED_RANKS = (1, 2, 3, 4)`) |
| **Memory Layout** | Alignment & strides | Dense, contiguous memory buffers (automatically enforced on export) |
| **Batch Convention** | Indexing | **Batch-first** convention; axis 0 is designated as the dynamic batch dimension |
| **Dynamic Axes** | Shape elasticity | Axis 0 symbolic name: `batch_size` (`-1` in `ModelInfo` / `IOTensorInfo`) |
| **Output Contracts** | Return topologies | Single tensor, tuple/list of tensors, and dictionary / `Mapping[str, Tensor]` |
| **Model Families** | Architectural scope | MLPs, CNNs (`Conv2d`), BERT / transformer encoder blocks, regression & classification heads |

---

## 2. Standardized Conversion Metadata & Contract

Pyreth enforces an explicit conversion contract prior to invoking the ONNX export backend:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    name: str  # ONNX graph input/output identifier
    shape: tuple[int | str, ...]  # Standardized shape (-1 for dynamic batch)
    dtype: str  # String representation ('float32', 'int64', etc.)


@dataclass(frozen=True)
class ConversionMetadata:
    model_name: str  # Base class name of the PyTorch module
    inputs: tuple[TensorSpec, ...]  # Input tensor specifications
    outputs: tuple[TensorSpec, ...]  # Output tensor specifications
    dynamic_batch: bool  # Whether axis 0 is dynamic
    dynamic_axes: dict[str, dict[int, str]]  # ONNX exporter dynamic axis map
    opset_version: int  # Targeted ONNX opset version
```

### Inspecting Models with `build_conversion_contract`

`pyreth.build_conversion_contract` validates model signatures, inspects input/output shapes, and standardizes metadata without writing files to disk:

```python
import torch
import pyreth


class VisionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(16, 10)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return {"logits": self.fc(h)}


model = VisionClassifier().eval()
sample = torch.randn(2, 3, 32, 32, dtype=torch.float32)

# Inspect contract
contract = pyreth.build_conversion_contract(
    model,
    sample,
    dynamic_batch=True,
    input_names=["image"],
)

print(
    contract.inputs[0]
)  # TensorSpec(name='image', shape=(-1, 3, 32, 32), dtype='float32')
print(contract.outputs[0])  # TensorSpec(name='logits', shape=(-1, 10), dtype='float32')
```

---

## 3. Failure Semantics & Boundary Validation

Pyreth validates invariants upfront and raises explicit exceptions:

| Condition | Exception | Failure Message Details |
| :--- | :--- | :--- |
| `model` is not an `nn.Module` | `TypeError` | `"model must be an instance of torch.nn.Module"` |
| `sample_inputs` is `None` or empty | `ValueError` | `"sample_inputs is required to export a PyTorch model to ONNX."` |
| Unsupported tensor dtype (e.g. `complex64`, `float64`) | `TypeError` | `"Unsupported dtype '<dtype>' for tensor '<name>'. Supported dtypes are: ['bool', 'float16', 'float32', 'int32', 'int64']"` |
| Unsupported tensor rank (0D scalar, 5D+ tensor) | `ValueError` | `"Unsupported rank <rank> for tensor '<name>'. Supported ranks are 1D, 2D, 3D, and 4D."` |
| `input_names` length mismatch | `ValueError` | `"input_names must match the number of input tensors; got X names for Y inputs"` |
| `output_names` length mismatch | `ValueError` | `"output_names must match the number of model outputs; got X names for Y outputs"` |
| Validation numerical divergence | `ValueError` | `"PyTorch reference output does not match the exported ONNX model output."` |
| Missing PyTorch dependency | `ModuleNotFoundError` | `"PyTorch support requires the optional 'torch' dependency."` |
| Missing ONNX dependency | `ModuleNotFoundError` | `"ONNX validation requires the optional 'onnx' dependency."` |

---

## 4. Export Utilities & Deterministic Graph Generation

`pyreth.export_torch_model_to_onnx` provides a deterministic export pipeline:
- **Evaluation Mode Safety**: Temporarily switches the model to `eval()` mode and restores previous training state on completion.
- **Deterministic Flags**: Sets `do_constant_folding=True`, `export_params=True`, and optional random seeds (`seed=...`).
- **Atomic File Writing**: Exports to a temporary staging file (`.tmp.onnx`) and performs an atomic replace into the destination path, preventing corrupted artifacts upon interrupted processes.
- **Custom Dynamic Axes**: `make_dynamic_axes(...)` or `dynamic_axes={...}` can be supplied to specify arbitrary dynamic dimensions (such as `sequence_length` in transformers) while keeping dynamic batch enabled on axis 0.

```python
from pyreth import export_torch_model_to_onnx, make_dynamic_axes

# Build dynamic axes for an NLP model with dynamic batch and dynamic sequence length
dynamic_axes = make_dynamic_axes(
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    additional_axes={
        "input_ids": {1: "sequence_length"},
        "attention_mask": {1: "sequence_length"},
    },
)

model_path = export_torch_model_to_onnx(
    model,
    (input_ids, attention_mask),
    "nlp_model.onnx",
    dynamic_axes=dynamic_axes,
    seed=42,
    validate=True,
)
```

---

## 5. Verification Helpers: Graph Integrity & Numerical Equivalence

Pyreth provides standalone diagnostic helpers to inspect exported models independently:

### `check_onnx_graph`
Validates graph protobuf integrity, imports, and shape inference:
```python
import pyreth

graph_info = pyreth.check_onnx_graph("nlp_model.onnx")
print(graph_info["opset_version"])  # 17
print(graph_info["input_names"])  # ['input_ids', 'attention_mask']
print(graph_info["output_names"])  # ['logits']
```

### `check_numerical_equivalence`
Executes sample inputs through both PyTorch reference and Pyreth `InferenceSession`, reporting maximum/mean absolute and relative discrepancies:
```python
metrics = pyreth.check_numerical_equivalence(
    model,
    "nlp_model.onnx",
    (input_ids, attention_mask),
    rtol=1e-5,
    atol=1e-5,
)
print("Max absolute error:", metrics["max_abs_diff"])
print("Mean absolute error:", metrics["mean_abs_diff"])
print("Max relative error:", metrics["max_rel_diff"])
```

---

## 6. Model Cache Management

Pyreth provides cache resolution utilities consistent with the benchmark suite:

```python
import pyreth

# Resolves default cache directory ($PYRETH_CACHE_DIR or benchmarks/.onnx_cache)
cache_dir = pyreth.get_default_cache_dir()

# Sanitize model identifiers into valid cache filenames
path = pyreth.resolve_cache_path("google/mobilenet_v2_1.0_224", prefix="image_cls__")
print(path)  # .../benchmarks/.onnx_cache/image_cls__google__mobilenet_v2_1.0_224.onnx

# Clear cached model files
pyreth.clear_cache(pattern="image_cls__*.onnx")
```

---

## 7. End-to-End Export & Direct Session Loading

Export a model to an ONNX file or load it directly into a Pyreth zero-copy `InferenceSession`:

```python
# Direct session instantiation from PyTorch module
session = pyreth.InferenceSession.from_torch(
    model,
    sample,
    output_path="models/classifier.onnx",
    dynamic_batch=True,
    validate=True,
)

# Run zero-copy inference with dynamic batch size
import numpy as np

batch_images = np.random.randn(8, 3, 32, 32).astype(np.float32)
outputs = session.run_numpy([batch_images])
print("Logits shape:", outputs[0].shape)  # (8, 10)
```
