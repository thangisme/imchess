import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("chess_eval.keras")

spec = (tf.TensorSpec((None, 8, 8, 13), tf.float32, name="board_input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="chess_eval.onnx"
)

print("ONNX model saved to chess_eval.onnx")
