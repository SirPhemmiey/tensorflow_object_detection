"""
This notebook will demontrate a pre-trained model to recognition plate number in an image.
Make sure to follow the [installation instructions](https://github.com/imamdigmi/plate-number-recognition#setup) before you start.
"""
import numpy as np
import os
from operator import itemgetter
from toolz import itertoolz
from imutils.video import FPS
import csv
import datetime
import six.moves.urllib as urllib
import sys
from itertools import groupby
from time import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from collections import defaultdict
from collections import Counter
from io import StringIO
from PIL import Image
import cv2

FILE_OUTPUT = './video_output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051.mp4')

# cap = cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051(online-video-cutter.com).mp4')

# cap = cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051_(online-video-cutter.com).mp4')


def inf(cap, detection_graph, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES): 
    
    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    csv_columns = ['start', 'end', 'prediction', 'probability', 'inference_time', 'bucket', 'truck', 'human', 'total_frames', 'total_cycles', 'avg_cycle_time', 'avg_fill_rate']
    
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
                    output_dict_array = []
                    objects = []
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    while(cap.isOpened()):
                        fps = cap.get(cv2.CAP_PROP_FPS) #frame per second
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        duration = current_frame/fps
                        minutes = int(duration / 60)
                        seconds = round(duration % 60)
                        length = f'{minutes}:{seconds}'
                        print(fps, current_frame, duration, length)
                                       
                        ret, frame = cap.read()
                        if ret == True:
                            # Get handles to input and output tensors
                            ops = tf.get_default_graph().get_operations()
                            all_tensor_names = {output.name for op in ops for output in op.outputs}
                            tensor_dict = {}
                            for key in [
                                'num_detections', 'detection_boxes', 'detection_scores',
                                'detection_classes', 'detection_masks'
                            ]:
                                tensor_name = key + ':0'
                                if tensor_name in all_tensor_names:
                                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                        tensor_name)
                            if 'detection_masks' in tensor_dict:
                                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    detection_masks, detection_boxes, frame.shape[0], frame.shape[1])
                                detection_masks_reframed = tf.cast(
                                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                                # Follow the convention by adding back the batch dimension
                                tensor_dict['detection_masks'] = tf.expand_dims(
                                    detection_masks_reframed, 0)
                            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                            # Actual Detection
                            start = time()
                            # Run inference
                            output_dict = sess.run(tensor_dict,
                                                feed_dict={image_tensor: np.expand_dims(frame, 0)})
                            end = time()
                            inference_time = end - start
                            
                            # all outputs are float32 numpy arrays, so convert types as appropriate
                            output_dict['num_detections'] = int(output_dict['num_detections'][0])
                            output_dict['detection_classes'] = output_dict[
                                'detection_classes'][0].astype(np.uint8)
                            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                            output_dict['detection_scores'] = output_dict['detection_scores'][0]
                            if 'detection_masks' in output_dict:
                                output_dict['detection_masks'] = output_dict['detection_masks'][0]

                            output_dict_array.append(output_dict)
                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                            frame,
                            np.squeeze(output_dict['detection_boxes']),
                            np.squeeze(output_dict['detection_classes']).astype(np.int32),
                            np.squeeze(output_dict['detection_scores']),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
                           
                            threshold = 0.3 # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects
                            # with open('data_.csv', 'a+') as csvfile:
                            #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            #     writer.writeheader()
                                # csvfile.write("start,end,prediction,probability,inference_time,humans,trucks,total_cycles,avg_cycle_time,avg_fill_rate\n")
                                
                            for index, value in enumerate(output_dict['detection_classes']):
                                object_dict = {}
                                if output_dict['detection_scores'][index] > threshold:
                                    prediction = (category_index.get(value)).get('name')
                                    probability = output_dict['detection_scores'][index]
                                    
                                    object_dict['start'] = length
                                    object_dict['end'] = length
                                    object_dict['prediction'] = (category_index.get(value)).get('name')
                                    object_dict['probability'] = output_dict['detection_scores'][index]
                                    object_dict['inference_time'] = inference_time
                                    # humans = 0 if (category_index.get(value)).get('name') != human else (category_index.get(value)).get('name').count()
                                    result_string = "{0},{1},{2},{3},{4},,,,,,\n".format(length,length,prediction,round(probability, 3),round(inference_time, 3))
                                    objects.append(object_dict)
                                    # objects.append(result_string)
                                    print('compiling....')  
                        else:
                            print('nananannana')
                            break   
                
                    arr = []
                    # Sum of all inference divided by the number of inference
                    for key, values in groupby(objects, lambda x: x['start'] and x['end']):
                        values = list(values)
                        val = {}
                        inference_times = []
                        probabilities = []
                        predictions = []
                        for val in list(values):
                            inference_times.append(float(val["inference_time"]))
                            probabilities.append(float(val["probability"]))
                            predictions.append(val["prediction"])
                        average_inference = float(float(sum(inference_times)) / len(inference_times))
                        average_probability = float(float(sum(probabilities)) / len(probabilities))
                        prediction_keys = Counter(predictions).keys()
                        prediction_values = Counter(predictions).values()
                        prediction_dictionary = dict(zip(prediction_keys, prediction_values))
                        if "bucket" not in prediction_keys:
                            prediction_dictionary.update(dict(bucket=0))
                        if "truck" not in prediction_keys:
                            prediction_dictionary.update(dict(truck=0))
                        if "human" not in prediction_keys:
                            prediction_dictionary.update(dict(human=0))
                        last_val = values[-1]
                        to_append = {"start": last_val.get("start"), "end": last_val.get("end"), "prediction": val["prediction"], "probability": average_probability,
                                    "inference_time": average_inference, "total_frames": total_frames}
                        to_append.update(prediction_dictionary)
                        arr.append(to_append)
                    print('correct', arr)

                    try:
                        with open('data_.csv', 'w+') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            writer.writeheader()
                            for data in arr:
                                writer.writerow(data)
                    except IOError:
                        print("I/O error")
                cap.release()
                cv2.destroyAllWindows()
                return output_dict_array
def keyfunc(x):
    return x['prediction']

def sort_uniq(sequence):
    return (x[0] for x in itertools.groupby(sorted(sequence)))

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

    # Model preparation
    MODEL_NAME = 'ssd_mobilenet_v2_quantized_300x300_coco'
    PATH_TO_CKPT = 'trained-inference-graphs/output_inference_graph_v2/frozen_inference_graph.pb'
    # PATH_TO_LABELS = os.path.join('data', '<LABEL_NAME>.pbtxt')
    PATH_TO_LABELS = 'annotations/label_map.pbtxt'
    NUM_CLASSES = 3
    TEST_IMAGE_PATHS = 'image_frames_'

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    inf(cap, detection_graph, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')

    # # Loading label map
    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)
    # csv_columns = ['start', 'end', 'prediction', 'probability', 'inference_time']

    # with detection_graph.as_default():
    #     with tf.Session(graph=detection_graph) as sess:
    #         # Definite input and output Tensors for detection_graph
    #         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            

    #         # Each box represents a part of the image where a particular object was detected.
    #         detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    #         # Each score represent how level of confidence for each of the objects.
    #         # Score is shown on the result image, together with the class label.
    #         detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    #         detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    #         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
    #         print('------------PRE----------')        
    #         while(cap.isOpened()):
                
    #             # Capture frame-by-frame
    #             ret, frame = cap.read()

    #             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    #             image_np_expanded = np.expand_dims(frame, 0)

    #             # Actual detection.
    #             start = time()
    #             #print('START TIME', start)
    #             (boxes, scores, classes, num) = sess.run(
    #                 [detection_boxes, detection_scores, detection_classes, num_detections],
    #                 feed_dict={image_tensor: image_np_expanded})
    #             end = time()
    #             #print('END TIME', end)
    #             inference_time = end - start
    #             #print('INFERENCE TIME', inference_time)
                
    #             print('------ACTUAL DETECTION-----')

    #             # Visualization of the results of a detection.
    #             vis_util.visualize_boxes_and_labels_on_image_array(
    #                 frame,
    #                 np.squeeze(boxes),
    #                 np.squeeze(classes).astype(np.int32),
    #                 np.squeeze(scores),
    #                 category_index,
    #                 use_normalized_coordinates=True,
    #                 line_thickness=8)

    #             objects = []
    #             threshold = 0.3 # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects
    #             for index, value in enumerate(classes[0]):
    #                 object_dict = {}
    #                 if scores[0, index] > threshold:
    #                     object_dict['start'] = start
    #                     object_dict['end'] = end
    #                     object_dict['prediction'] = (category_index.get(value)).get('name')
    #                     object_dict['probability'] = scores[0, index]
    #                     object_dict['inference_time'] = inference_time
    #                     objects.append(object_dict)
    #             print ('Objects>>', objects)
    #         csv_file = "data.csv"
    #         print('OUTSIDE LOOP!')
    #         try:
    #             with open(csv_file, 'w') as csvfile:
    #                 writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #                 writer.writeheader()
    #                 for data in objects:
    #                     writer.writerow(data)
    #         except IOError:
    #             print("I/O error")
    #             # try:
    #             #     with open(csv_file, 'a') as csvfile:
    #             #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #             #         writer.writeheader()
    #             #         for data in objects:
    #             #             writer.writerow(data)
    #             # except IOError:
    #             #     print("I/O error")
                
    #             # if ret == True:
    #             #     # Saves for video
    #             #     #out.write(frame)

    #             #     # Display the resulting frame
    #             #     #cv2.imshow('Charving Detection', frame)

    #             #     #Close window when "Q" button pressed
    #             #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #             #         break
    #             # else:
    #             #     break
    #     # When everything done, release the video capture and video write objects
    #     cap.release()
    #     # Closes all the frames
    #     cv2.destroyAllWindows()
    # #return objects

pipeline(cap)