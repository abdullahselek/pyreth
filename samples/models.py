"""Pure-Python ONNX model builders for Pyreth sample scripts.

Constructs valid minimal ONNX protobuf byte buffers in memory without requiring
external `onnx` or `protobuf` packages, so all samples run out-of-the-box.
"""

import struct

import numpy as np


def _encode_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_varint_field(field_number: int, val: int) -> bytes:
    return _encode_tag(field_number, 0) + _encode_varint(val)


def _encode_bytes_field(field_number: int, data: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_string_field(field_number: int, text: str) -> bytes:
    return _encode_bytes_field(field_number, text.encode("utf-8"))


def _encode_value_info(name: str, elem_type: int, dims: list[int | str]) -> bytes:
    shape_bytes = bytearray()
    for d in dims:
        if isinstance(d, str):
            dim_proto = _encode_string_field(2, d)
        else:
            dim_proto = _encode_varint_field(1, d)
        shape_bytes += _encode_bytes_field(1, dim_proto)

    tensor_type = _encode_varint_field(1, elem_type) + _encode_bytes_field(
        2, bytes(shape_bytes)
    )
    type_proto = _encode_bytes_field(1, tensor_type)
    return _encode_string_field(1, name) + _encode_bytes_field(2, type_proto)


def _encode_node(
    inputs: list[str], outputs: list[str], name: str, op_type: str
) -> bytes:
    buf = bytearray()
    for i in inputs:
        buf += _encode_string_field(1, i)
    for o in outputs:
        buf += _encode_string_field(2, o)
    buf += _encode_string_field(3, name)
    buf += _encode_string_field(4, op_type)
    return bytes(buf)


def _encode_float_initializer(
    name: str, dims: list[int], values: list[float] | np.ndarray
) -> bytes:
    buf = bytearray()
    for d in dims:
        buf += _encode_varint_field(1, d)
    buf += _encode_varint_field(2, 1)  # TensorProto.DataType.FLOAT = 1
    buf += _encode_string_field(8, name)
    if isinstance(values, np.ndarray):
        raw_floats = np.asarray(values, dtype="<f4").tobytes()
    else:
        raw_floats = struct.pack(f"<{len(values)}f", *values)
    buf += _encode_bytes_field(9, raw_floats)
    return bytes(buf)


def _encode_string_entry(key: str, val: str) -> bytes:
    return _encode_string_field(1, key) + _encode_string_field(2, val)


def build_mlp_onnx_bytes() -> bytes:
    """Build `y = Relu(MatMul(x, W) + b)` with `x: [batch_size, 4] -> y: [batch_size, 3]`."""
    w_vals = [
        1.0,
        -1.0,
        0.5,
        0.0,
        2.0,
        -0.5,
        2.0,
        0.0,
        1.0,
        -1.0,
        1.0,
        2.0,
    ]
    b_vals = [0.5, -1.0, 0.0]

    graph = bytearray()
    graph += _encode_bytes_field(
        1, _encode_node(["x", "W"], ["xw"], "mm_node", "MatMul")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["xw", "b"], ["xw_b"], "add_node", "Add")
    )
    graph += _encode_bytes_field(1, _encode_node(["xw_b"], ["y"], "relu_node", "Relu"))
    graph += _encode_string_field(2, "mlp_linear_relu_graph")
    graph += _encode_bytes_field(5, _encode_float_initializer("W", [4, 3], w_vals))
    graph += _encode_bytes_field(5, _encode_float_initializer("b", [3], b_vals))
    graph += _encode_bytes_field(11, _encode_value_info("x", 1, ["batch_size", 4]))
    graph += _encode_bytes_field(12, _encode_value_info("y", 1, ["batch_size", 3]))

    opset = _encode_string_field(1, "") + _encode_varint_field(2, 17)

    model = bytearray()
    model += _encode_varint_field(1, 9)  # ir_version = 9
    model += _encode_string_field(2, "pyreth_samples")
    model += _encode_string_field(4, "ai.pyreth.samples")
    model += _encode_varint_field(5, 1)  # model_version = 1
    model += _encode_string_field(6, "Sample Linear + ReLU MLP model for Pyreth")
    model += _encode_bytes_field(7, bytes(graph))
    model += _encode_bytes_field(8, opset)
    model += _encode_bytes_field(14, _encode_string_entry("architecture", "mlp_4x3"))
    model += _encode_bytes_field(14, _encode_string_entry("task", "classification"))
    return bytes(model)


def expected_mlp_numpy(x: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation of `build_mlp_onnx_bytes()`."""
    w = np.array(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -0.5],
            [2.0, 0.0, 1.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    b = np.array([0.5, -1.0, 0.0], dtype=np.float32)
    return np.maximum(0.0, x @ w + b)


def build_dual_io_onnx_bytes(features: int = 8) -> bytes:
    """Build a multi-input, multi-output graph: `sum = a + b`, `prod = a * b`."""
    graph = bytearray()
    graph += _encode_bytes_field(
        1, _encode_node(["a", "b"], ["sum"], "add_node", "Add")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["a", "b"], ["prod"], "mul_node", "Mul")
    )
    graph += _encode_string_field(2, "dual_io_arithmetic_graph")
    graph += _encode_bytes_field(
        11, _encode_value_info("a", 1, ["batch_size", features])
    )
    graph += _encode_bytes_field(
        11, _encode_value_info("b", 1, ["batch_size", features])
    )
    graph += _encode_bytes_field(
        12, _encode_value_info("sum", 1, ["batch_size", features])
    )
    graph += _encode_bytes_field(
        12, _encode_value_info("prod", 1, ["batch_size", features])
    )

    opset = _encode_string_field(1, "") + _encode_varint_field(2, 17)
    model = bytearray()
    model += _encode_varint_field(1, 9)
    model += _encode_string_field(2, "pyreth_samples")
    model += _encode_bytes_field(7, bytes(graph))
    model += _encode_bytes_field(8, opset)
    return bytes(model)


def build_serving_mlp_onnx_bytes(
    in_features: int = 64, hidden_features: int = 128, out_features: int = 16
) -> bytes:
    """Build a 2-layer MLP ONNX model: `Y = Relu(X @ W1 + B1) @ W2 + B2`."""
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal((in_features, hidden_features), dtype=np.float32) * 0.1
    b1 = np.zeros((hidden_features,), dtype=np.float32)
    w2 = rng.standard_normal((hidden_features, out_features), dtype=np.float32) * 0.1
    b2 = np.zeros((out_features,), dtype=np.float32)

    graph = bytearray()
    graph += _encode_string_field(2, "serving_two_layer_mlp")
    graph += _encode_bytes_field(
        5, _encode_float_initializer("W1", [in_features, hidden_features], w1)
    )
    graph += _encode_bytes_field(
        5, _encode_float_initializer("B1", [hidden_features], b1)
    )
    graph += _encode_bytes_field(
        5, _encode_float_initializer("W2", [hidden_features, out_features], w2)
    )
    graph += _encode_bytes_field(5, _encode_float_initializer("B2", [out_features], b2))

    graph += _encode_bytes_field(
        1, _encode_node(["x", "W1"], ["h_mm"], "mm1", "MatMul")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["h_mm", "B1"], ["h_add"], "add1", "Add")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["h_add"], ["h_relu"], "relu1", "Relu")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["h_relu", "W2"], ["y_mm"], "mm2", "MatMul")
    )
    graph += _encode_bytes_field(1, _encode_node(["y_mm", "B2"], ["y"], "add2", "Add"))

    graph += _encode_bytes_field(
        11, _encode_value_info("x", 1, ["batch_size", in_features])
    )
    graph += _encode_bytes_field(
        12, _encode_value_info("y", 1, ["batch_size", out_features])
    )

    opset = _encode_string_field(1, "") + _encode_varint_field(2, 17)
    model = bytearray()
    model += _encode_varint_field(1, 9)
    model += _encode_string_field(2, "pyreth_samples")
    model += _encode_varint_field(5, 1)
    model += _encode_bytes_field(7, bytes(graph))
    model += _encode_bytes_field(8, opset)
    return bytes(model)
