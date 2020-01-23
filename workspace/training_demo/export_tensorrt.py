

# /Users/akindeoluwafemi/Documents/works/ML/tensorflow_3/workspace/training_demo/trained-inference-graphs/output_inference_graph_v2/frozen_inference_graph.pb
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
with tf.Session() as sess:
    # First deserialize your frozen graph:
    input_path = "/Users/akindeoluwafemi/Documents/works/ML/tensorflow_3/workspace/training_demo/trained-inference-graphs/output_inference_graph_v2/frozen_inference_graph.pb"
    with tf.gfile.GFile(input_path, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    # Now you can create a TensorRT inference graph from your
    # frozen graph:
    converter = trt.TrtGraphConverter(
	    input_graph_def=frozen_graph,
	    nodes_blacklist=['logits', 'classes']) #output nodes
    trt_graph = converter.convert()
    # Import the TensorRT graph into a new graph and run:
    output_node = tf.import_graph_def(
        trt_graph,
        return_elements=['logits', 'classes'])
    sess.run(output_node)