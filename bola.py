# Ignore warnings
import warnings
warnings.filterwarnings("ignore") # Warning will make operation confuse!!!

# Model
import torch
model = torch.hub.load(   'yolov8' # Use backend for yolov5 in this folder
                        , 'custom' # to use model in this folder
                        , path='best (4).pt' # the name of model is this folder
                        , source='local' # to use backend from this folder
                        , force_reload=True # clear catch
                        , device = 'cpu' # I want to use CPU
                    ) 
model.conf = 0.75 # NMS confidence threshold
model.iou = 0.15  # IoU threshold
model.multi_label = True  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [0,1]   # (optional list) filter by class, 77 for "tendy bear"
# I suggest to see the label list in https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

# RUN
import cv2
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture("Video/Merah_Hijau.mp4") # Read USB Camera
while(video.isOpened()):
    # Read Frame
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    # Object Detection
    results = model(frame)
    cv2.imshow('Object detector', results.render()[0])
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'): break

# Clean up
video.release()
cv2.destroyAllWindows()