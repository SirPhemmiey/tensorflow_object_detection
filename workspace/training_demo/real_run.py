"""
This notebook will demontrate a pre-trained model to recognition plate number in an image.
Make sure to follow the [installation instructions](https://github.com/imamdigmi/plate-number-recognition#setup) before you start.
"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2 as cv2

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

FILE_OUTPUT = './video_output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

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

# Model preparation
MODEL_NAME = 'ssd_mobilenet_v2_quantized_300x300_coco'
PATH_TO_CKPT = 'trained-inference-graphs/output_inference_graph_v2/frozen_inference_graph.pb'
# PATH_TO_LABELS = os.path.join('data', '<LABEL_NAME>.pbtxt')
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
NUM_CLASSES = 3

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
        
        print('------------PRE----------')
        print(detection_scores, num_detections, detection_classes)
        

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)
            
            print('--------INSIDE WHILE LOOP-----')
            # print(frame)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            if ret == True:
                # Saves for video
                out.write(frame)

                # Display the resulting frame
                cv2.imshow('Charving Detection', frame)

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
    
    
    
    
    """
This notebook will demontrate a pre-trained model to recognition plate number in an image.
Make sure to follow the [installation instructions](https://github.com/imamdigmi/plate-number-recognition#setup) before you start.
"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
from time import time
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

FILE_OUTPUT = './video_output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051.mp4')

def pipeline(cap) :
    
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
            
            print('------------PRE----------')        
            while(cap.isOpened()):
                
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)

                # Actual detection.
                start = time()
                #print('START TIME', start)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                end = time()
                #print('END TIME', end)
                inference_time = end - start
                #print('INFERENCE TIME', inference_time)
                
                print('------ACTUAL DETECTION-----')
                # print('image exp>', image_np_expanded)
                # print('boxes', np.squeeze(boxes))
                # print('scores', np.squeeze(scores))
                # print('classes', np.squeeze(classes))
                # print('num', np.squeeze(num))
                # Here output the category as string and score to terminal
                #print([category_index.get(i) for i in classes[0]])
                # print(scores)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                objects = []
                threshold = 0.3 # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > threshold:
                        # object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                        #             scores[0, index]
                        object_dict['start'] = start
                        object_dict['end'] = end
                        object_dict['prediction'] = (category_index.get(value)).get('name')
                        object_dict['probability'] = scores[0, index]
                        object_dict['inference_time'] = inference_time
                        
                        # print('NAME1>>>', (category_index.get(value)))
                        # print('NAME2>>>', (category_index.get(value)).get('name'))
                        objects.append(object_dict)
                print ('Objects>>', objects)
        
                #print('OKAY???', len(np.where(scores[0] > threshold)[0])/num_detections[0])

                # if ret == True:
                #     # Saves for video
                #     #out.write(frame)

                #     # Display the resulting frame
                #     #cv2.imshow('Charving Detection', frame)

                #     #Close window when "Q" button pressed
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                # else:
                #     break
            
                # end = time()
                # print('end>>', end)
        # When everything done, release the video capture and video write objects
        cap.release()
    # out.release()

        # Closes all the frames
        cv2.destroyAllWindows()