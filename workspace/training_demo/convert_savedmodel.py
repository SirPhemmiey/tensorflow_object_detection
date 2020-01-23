import tensorflow as tf

# Construct a basic model.
# root = tf.train.Checkpoint()
# root.v1 = tf.Variable(3.)
# root.v2 = tf.Variable(2.)
# root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# # Save the model.
export_dir = "trained-inference-graphs/output_inference_graph_v1.pb/saved_model"
# input_data = tf.constant(1., shape=[1, 1])
# to_save = root.f.get_concrete_function(input_data)
# tf.saved_model.save(root, export_dir, to_save)

# Convert the model.
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()
# print(tflite_model)


# model = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# concrete_func = model.signatures[
# tf.from_saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# concrete_func.inputs[0].set_shape([1, 256, 256, 3])
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()


converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(export_dir)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
tflite_model = converter.convert()