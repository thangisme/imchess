import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

try:
    import tf2onnx
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False

def convert_keras_to_onnx(input_model_path, output_model_path, input_shape=(8, 8, 13), opset=13):
    
    if not TF2ONNX_AVAILABLE:
        print("ERROR: tf2onnx package not found. Please install with: pip install tf2onnx")
        return False
    
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model file not found: {input_model_path}")
        return False
    
    try:
        print(f"Loading Keras model from {input_model_path}...")
        model = keras.models.load_model(input_model_path)
        print("Model loaded successfully.")
        
        os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
        
        spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="board_input"),)
        
        print(f"Converting model to ONNX (opset {opset})...")
        model_proto, _ = tf2onnx.convert.from_keras(
            model, 
            input_signature=spec, 
            opset=opset, 
            output_path=output_model_path
        )
        
        if os.path.exists(output_model_path):
            print(f"ONNX model successfully saved to {output_model_path}")
            return True
        else:
            print("ERROR: ONNX conversion completed but output file was not created")
            return False
            
    except Exception as e:
        print(f"ERROR during ONNX conversion: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your TensorFlow and tf2onnx versions are compatible")
        print("2. Try updating tf2onnx: pip install -U tf2onnx")
        print("3. Check for model compatibility issues (custom layers, etc.)")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to ONNX format")
    parser.add_argument("--input", type=str, required=True, help="Path to input Keras model file (.keras)")
    parser.add_argument("--output", type=str, default=None, help="Path for output ONNX model file (.onnx)")
    parser.add_argument("--height", type=int, default=8, help="Model input height")
    parser.add_argument("--width", type=int, default=8, help="Model input width")
    parser.add_argument("--channels", type=int, default=13, help="Model input channels")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".onnx"
    
    input_shape = (args.height, args.width, args.channels)
    print(f"Using input shape: {input_shape}")
    
    success = convert_keras_to_onnx(args.input, args.output, input_shape, args.opset)
    
    if success:
        print("\nConversion completed successfully!")
        sys.exit(0)
    else:
        print("\nConversion failed. See error messages above.")
        sys.exit(1)

