
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2



FILE_OUTPUT = './video_output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

INPUT_FILE='/Users/akindeoluwafemi/Downloads/051.mp4'
# Playing video from file
cap = cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051.mp4')

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print('-------SIZES-----')
print(frame_width, frame_height)

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))

sys.path.append("..")

# Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def get_tensor_from_video(input_file, frame):
    vidcap = cv2.VideoCapture(input_file)
    vidcap.set(1, frame)
    
    success,image = vidcap.read()
    if success: 
        tensor = read_tensor_from_image_file(image, 640, 480)
        res = sess.run(tensor)
        return res
    else:
        return []

def read_tensor_from_image_file(image,
                                input_height=480,
                                input_width=640,
                                input_mean=0,
                                input_std=255):
    img = cv2.resize(image, (input_width,input_height), interpolation = cv2.INTER_AREA)
    float_caster = tf.cast(img, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    #normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
    return dims_expander

# Model preparation
MODEL_NAME = 'ssd_mobilenet_v2_quantized_300x300_coco'
PATH_TO_CKPT = 'trained-inference-graphs/output_inference_graph_v2/frozen_inference_graph.pb'
# PATH_TO_LABELS = os.path.join('data', '<LABEL_NAME>.pbtxt')
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
NUM_CLASSES = 3
TEST_IMAGE_PATHS = 'image_frames_'

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')        

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            get_tensor_from_video(INPUT_FILE, frame)

            if ret == True:
                # Saves for video
                #out.write(frame)

                # Display the resulting frame
                #cv2.imshow('Charving Detection', frame)

                #Close window when "Q" button pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


