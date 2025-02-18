import os
import warnings
from typing import Optional, Union
import tensorflow as tf
import tempfile

from larq_compute_engine.mlir._tf_tfl_flatbuffer import (
    convert_graphdef_to_tflite_flatbuffer,
    convert_saved_model_to_tflite_flatbuffer,
)

from larq_compute_engine.mlir.python.util import modify_integer_quantized_model_io_type

from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.lite.python.util import get_tensor_name
from tensorflow.python.eager import def_function
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def concrete_function_from_keras_model(model):
    if not model.inputs:
        raise ValueError("El modelo no tiene entradas definidas.")
    # Si el modelo tiene una única entrada:
    if len(model.inputs) == 1:
        input_shape = model.inputs[0].shape
        input_dtype = model.inputs[0].dtype
        # Se mantiene el batch dimension definido en el modelo.
        input_spec = tf.TensorSpec(input_shape, input_dtype)
    else:
        # Para modelos con múltiples entradas, se genera una lista de TensorSpec.
        input_spec = [
            tf.TensorSpec(tensor.shape, tensor.dtype) for tensor in model.inputs
        ]

    # Obtiene la función concreta a partir del input_spec.
    concrete_func = model.get_concrete_function(input_spec)
    return concrete_func


def _contains_training_quant_op(graph_def):
    """Checks if the graph contains any training-time quantization ops."""
    training_quant_ops = {
        "FakeQuantWithMinMaxVars",
        "FakeQuantWithMinMaxVarsPerChannel",
        "FakeQuantWithMinMaxArgs",
        "FakeQuantWithMinMaxArgsPerChannel",
        "QuantizeAndDequantizeV2",
        "QuantizeAndDequantizeV3",
    }

    for node_def in graph_def.node:
        if node_def.op in training_quant_ops:
            return True
    for function in graph_def.library.function:
        for node_def in function.node_def:
            if node_def.op in training_quant_ops:
                return True
    return False


def _validate_options(
    *,
    inference_input_type=None,
    inference_output_type=None,
    target=None,
    experimental_default_int8_range=None,
):
    if inference_input_type not in (tf.float32, tf.int8):
        raise ValueError(
            "Expected `inference_input_type` to be either `tf.float32` or `tf.int8`, "
            f"got {inference_input_type}."
        )
    if inference_output_type not in (tf.float32, tf.int8):
        raise ValueError(
            "Expected `inference_output_type` to be either `tf.float32` or `tf.int8`, "
            f"got {inference_output_type}."
        )
    if target not in ("arm", "xcore"):
        raise ValueError(f'Expected `target` to be "arm" or "xcore", but got {target}.')

    if not tf.executing_eagerly():
        raise RuntimeError(
            "Graph mode is not supported. Please enable eager execution using "
            "tf.enable_eager_execution() when using TensorFlow 1.x"
        )
    if experimental_default_int8_range:
        warnings.warn(
            "Using `experimental_default_int8_range` as fallback quantization stats. "
            "This should only be used for latency tests."
        )


def convert_saved_model(
    saved_model_dir: Union[str, os.PathLike],
    *,  # Require remaining arguments to be keyword-only.
    inference_input_type: tf.DType = tf.float32,
    inference_output_type: tf.DType = tf.float32,
    target: str = "arm",
    experimental_default_int8_range: Optional[tuple[float, float]] = None,
) -> bytes:
    """Converts a SavedModel to TFLite flatbuffer.

    !!! example
        ```python
        tflite_model = convert_saved_model(saved_model_dir)
        with open("/tmp/my_model.tflite", "wb") as f:
            f.write(tflite_model)
        ```

    # Arguments
        saved_model_dir: SavedModel directory to convert.
        inference_input_type: Data type of the input layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        inference_output_type: Data type of the output layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        target: Target hardware platform. Defaults to "arm", must be either "arm"
            or "xcore".
        experimental_default_int8_range: Tuple of integers representing `(min, max)`
            range values for all arrays without a specified range. Intended for
            experimenting with quantization via "dummy quantization". (default None)

    # Returns
        The converted data in serialized format.
    """
    _validate_options(
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        target=target,
        experimental_default_int8_range=experimental_default_int8_range,
    )

    saved_model_dir = str(saved_model_dir)
    saved_model_tags = [tf.saved_model.SERVING]
    saved_model_exported_names = [tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    from tensorflow.python.saved_model import loader_impl

    saved_model_pb, _ = loader_impl.parse_saved_model_with_debug_info(saved_model_dir)

    saved_model_version = saved_model_pb.saved_model_schema_version
    if saved_model_version not in (1, 2):
        raise ValueError(
            f"SavedModel file format({saved_model_version}) is not supported"
        )

    tflite_buffer = convert_saved_model_to_tflite_flatbuffer(
        saved_model_dir,
        saved_model_tags,
        saved_model_exported_names,
        saved_model_version,
        target,
        experimental_default_int8_range,
    )

    if inference_input_type != tf.float32 or inference_output_type != tf.float32:
        tflite_buffer = modify_integer_quantized_model_io_type(
            tflite_buffer,
            inference_input_type=inference_input_type,
            inference_output_type=inference_output_type,
        )

    return tflite_buffer


def convert_keras_model(
    model: tf.keras.Model,
    *,  # Require remaining arguments to be keyword-only.
    inference_input_type: tf.DType = tf.float32,
    inference_output_type: tf.DType = tf.float32,
    target: str = "arm",
    experimental_default_int8_range: Optional[tuple[float, float]] = None,
) -> bytes:
    """Converts a Keras model to TFLite flatbuffer.

    !!! example
        ```python
        tflite_model = convert_keras_model(model)
        with open("/tmp/my_model.tflite", "wb") as f:
            f.write(tflite_model)
        ```

    # Arguments
        model: The model to convert.
        inference_input_type: Data type of the input layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        inference_output_type: Data type of the output layer. Defaults to `tf.float32`,
            must be either `tf.float32` or `tf.int8`.
        target: Target hardware platform. Defaults to "arm", must be either "arm"
            or "xcore".
        experimental_default_int8_range: Tuple of integers representing `(min, max)`
            range values for all arrays without a specified range. Intended for
            experimenting with quantization via "dummy quantization". (default None)

    # Returns
        The converted data in serialized format.
    """
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            f"Expected `model` argument to be a `tf.keras.Model` instance, got `{model}`."
        )
    print('Es un modelo de keras')
    if hasattr(model, "dtype_policy") and model.dtype_policy.name != "float32":
        raise ValueError(
            "Mixed precision float16 models are not supported by the TFLite converter, "
            "please convert them to float32 first. See also: "
            "https://github.com/tensorflow/tensorflow/issues/46380"
        )
    print('Se acepto la policy')
    _validate_options(
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        target=target,
        experimental_default_int8_range=experimental_default_int8_range,
    )
    print('paso la validacion')
    # First attempt conversion as saved model
    try:
        with tempfile.TemporaryDirectory() as saved_model_dir:
            model.save(saved_model_dir, save_format="tf")
            print('Guardado el modelo')
            return convert_saved_model(
                saved_model_dir,
                inference_input_type=inference_input_type,
                inference_output_type=inference_output_type,
                experimental_default_int8_range=experimental_default_int8_range,
                target=target,
            )
        print('Se convirtio el modelo')
    except Exception as e:
        print('Error al convertir el modelo', e)
        warnings.warn(
            "Saved-model conversion failed, falling back to graphdef-based conversion."
        )
    func = concrete_function_from_keras_model(model)
    print('Se obtuvo la funcion concreta')
    frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
    print('Se convirtio a constantes')
    input_tensors = [
        tensor for tensor in frozen_func.inputs if tensor.dtype != tf.dtypes.resource
    ]
    print('Se obtuvieron los tensores de entrada')
    output_tensors = frozen_func.outputs
    print('Se obtuvieron los tensores de salida')
    graph_def = frozen_func.graph.as_graph_def()
    print('Se obtuvo el grafo')
    should_quantize = (
        _contains_training_quant_op(graph_def)
        or experimental_default_int8_range is not None
    )
    print('Se debe cuantizar')

    # Checks dimensions in input tensor.
    for tensor in input_tensors:
        # Note that shape_list might be empty for scalar shapes.
        shape_list = tensor.shape.as_list()
        if None in shape_list[1:]:
            raise ValueError(
                "None is only supported in the 1st dimension. Tensor '{0}' has "
                "invalid shape '{1}'.".format(get_tensor_name(tensor), shape_list)
            )
        elif shape_list and shape_list[0] is None:
            # Set the batch size to 1 if undefined.
            shape = tensor.shape.as_list()
            shape[0] = 1
            tensor.set_shape(shape)

    tflite_buffer = convert_graphdef_to_tflite_flatbuffer(
        graph_def.SerializeToString(),
        [get_tensor_name(tensor) for tensor in input_tensors],
        [DataType.Name(tensor.dtype.as_datatype_enum) for tensor in input_tensors],
        [tensor.shape.as_list() for tensor in input_tensors],
        [get_tensor_name(tensor) for tensor in output_tensors],
        should_quantize,
        target,
        experimental_default_int8_range,
    )

    if should_quantize and (
        inference_input_type != tf.float32 or inference_output_type != tf.float32
    ):
        tflite_buffer = modify_integer_quantized_model_io_type(
            tflite_buffer,
            inference_input_type=inference_input_type,
            inference_output_type=inference_output_type,
        )

    return tflite_buffer
