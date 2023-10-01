
import numpy as np
import cv2
from ultralytics import YOLO
import math

# # opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = [[0,255,0],[0,0,255]]
# for i in range(len(class_list)):
#     r = random.randint(0, 255)
#     g = random.randint(0, 255)
#     b = random.randint(0, 255)
#     detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("29-09v7.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 640
hj1 = 0
hj2 = 0 
mh1 = 0
mh2 = 0
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture("..\Video\Merah_Hijau.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
# fourcc=cv2.VideoWriter_fourcc(*'XVID')
# out=cv2.VideoWriter('new.avi',fourcc,20.0,(320,320))
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (frame_wid, frame_hyt))
    # contrast = 0.3
    # brightness = 0
    # frame = np.clip(contrast * frame + brightness, 0, 80)

    #garis tengah kamera Horizontal
    cv2.line(frame, (0,320), (640,320), (255, 0, 0), 1)
    #garis tengah kamera Vertikal
    cv2.line(frame, (320,0), (320,640), (255, 0, 0), 3)
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.25, save=True, imgsz=320)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    # print(DP)
    
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)
            
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # cv2.rectangle(
            #     frame,
            #     (int(bb[0]), int(bb[1])),
            #     (int(bb[2]), int(bb[3])),
            #     detection_colors[int(clsID)],
            #     3,
            # )
            
            radiusc = math.sqrt(int((bb[0]-bb[2])**2)+int((bb[1]-bb[3])**2))
            cv2.circle(
                frame,
                (int((bb[0] +  bb[2])/2),int((bb[1] + bb[3])/2)),
                int(radiusc/3),
                detection_colors[int(clsID)],
                3,
            )
            
            if int(clsID) == 0:
                # global center1
                hj1=int((bb[0] + bb[2])/2)
                hj2=int((bb[1] + bb[3])/2)
                center1 = hj1,hj2
                cv2.circle(frame, center1, 1, (0, 255, 0), 5)
                
                
            elif int(clsID) == 1:
                # global center2
                mh1 = int((bb[0] + bb[2])/2)
                mh2 = int((bb[1] + bb[3])/2)
                center2 = mh1,mh2
                cv2.circle(frame, center2, 1, (0, 0, 255), 5)
           
            cc1 = (hj1,hj2)
            cc2 = (mh1,mh2)
            a = (mh1+hj1)/2
            b = (hj2+mh2)/2
            v = int(a),int(b)
            #garis 2 titik
            cv2.line(frame, cc1, cc2, (0, 255, 255), 2)
           #titik tengah 2 bola
            cv2.circle(frame,v, 1, (255,255,255), 10)
            
            
            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)]
                + " "
                + str(round(conf, 3))
                + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
