!pip install tf2onnx
!pip install onnx
!pip install tensorflow

import tensorflow as tf

# Reuse the loaded model from before (or reload if needed)
model = tf.keras.models.load_model(
    '/kaggle/input/3dcae-model-project/tensorflow1/default/1/3DCAE_model_to_be_used.h5',
    compile=False
)
model.compile(optimizer='adam', loss='mse')

print("Model ready for Flex-enabled conversion...")

# Converter with Flex (SELECT_TF_OPS) for MaxPool3D fallback
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Native ops where possible
    tf.lite.OpsSet.SELECT_TF_OPS     # Flex for tf.MaxPool3D
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize for speed
converter.allow_custom_ops = True  # Required for Flex

try:
    tflite_model = converter.convert()
    print("✅ Flex TFLite conversion successful!")
    
    # Save
    with open('/kaggle/working/new_3DCAE_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model size: {len(tflite_model) / (1024*1024):.1f} MB")
    
    # Quick inference test (dummy input)
    interpreter = tf.lite.Interpreter(model_path='/kaggle/working/new_3DCAE_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Dummy batch=1 input (normalize if your data needs it)
    dummy_input = tf.random.normal([1, 16, 128, 128, 3]).numpy()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"✅ Dummy inference works! Output shape: {output.shape}, min/max: {output.min():.3f}/{output.max():.3f}")
    
except Exception as e:
    print(f"❌ Still failed: {e}")