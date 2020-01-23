import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='trained-inference-graphs/output_tflite/tflite_graph.pb'
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def =  tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        
LOGDIR='/logs/tests/1/'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()