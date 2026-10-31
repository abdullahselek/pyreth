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
    name: str                       # ONNX graph input/output identifier
    shape: tuple[int | str, ...]    # Standardized shape (-1 for dynamic batch)
    dtype: str                      # String representation ('float32', 'int64', etc.)

@dataclass(frozen=True)
class ConversionMetadata:
    model_name: str                 # Base class name of the PyTorch module
    inputs: tuple[TensorSpec, ...]  # Input tensor specifications
    outputs: tuple[TensorSpec, ...] # Output tensor specifications
    dynamic_batch: bool             # Whether axis 0 is dynamic
    dynamic_axes: dict[str, dict[int, str]] # ONNX exporter dynamic axis map
    opset_version: int              # Targeted ONNX opset version
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

print(contract.inputs[0])   # TensorSpec(name='image', shape=(-1, 3, 32, 32), dtype='float32')
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

## 4. End-to-End Export & Direct Session Loading

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
