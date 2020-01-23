
# import cv2
 
# # Opens the Video file
# cap= cv2.VideoCapture('/Users/akindeoluwafemi/Downloads/051.mp4')
# count=0
# i=0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     print("FRAME", frame)
#     if ret:
#         cv2.imwrite('.video_frames/frame-'+str(count)+'.jpg', frame)
#         count += 30 #here you can enter every nth frame you want to capture
#         cap.set(1, count)
#     else:
#         break
 
# cap.release()
# cv2.destroyAllWindows()


import cv2
import os

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, 'frame-%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    print('done')

video_to_frames('/Users/akindeoluwafemi/Downloads/051.mp4', './image_frames')
