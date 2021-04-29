# yoloface

[Colab](https://colab.research.google.com/drive/1VVrLoOpTvtCiS3qj_Re-7b6Da9SotWyH?usp=sharing)

### This is simpleste way to detect face:
```bash
You only look once (YOLO) is a state-of-the-art, real-time object detection system. It is based on Deep Learning.
Face detection is one of the important tasks of object detection. We apply a single neural network to the full image.This project focuses on improving the accuracy of detecting the face using the model of deep learning network (YOLO).This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

```

## User Installation :
If you already have a working installation of numpy and pandas, opencv the easiest way to install yoloface is using pip
```bash
pip install yoloface
```



## This Package Depend On Other Packages:
```bash
# Prerequisites
  1.numpy
  2.cv2
  3.os 
  4.PIL
  5.gdown
  6.time
  7.IPython
```

# Usage

## Face Detection In Image
```python

# import libraries
from yoloface import face_analysis
import numpy
import cv2


face=face_analysis()    #  Auto Download a large weight files from Google Drive.
                        #  only first time.
                        #  Automatically  create folder .yoloface on cwd.

# example 1
%%time
img,box,conf=face.face_detection(image_path='path/to/jpg/or/png/filename.jpg',model='tiny')
print(box)
print(conf)
face.show_output(img,box)



# example 2
%%time
img,box,conf=face.face_detection(image_path='path/to/jpg/or/png/filename.jpg',model='full')
print(box)
print(conf)
face.show_output(img,box)

```
# Real-Time Detection on a Webcam
```python
from yoloface import face_analysis
import numpy
import cv2

# example 3
cap = cv2.VideoCapture(0)
while True: 
    _, frame = cap.read()
    _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    output_frame=face.show_output(frame,box,frame_status=True)
    cv2.imshow('frame',output_frame)
    key=cv2.waitKey(1)
    if key ==ord('v'): 
        break 
cap.release()
cv2.destroyAllWindows()
#press v (exits)



# example 4
cap = cv2.VideoCapture('')
while True: 
    _, frame = cap.read()
    __,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='full')
    output_frame=face.show_output(img=frame,face_box=box,frame_status=True)
    print(box)
    cv2.imshow('frame',output_frame)
    key=cv2.waitKey(0)
    if key ==ord('v'): 
        break 
cap.release()
cv2.destroyAllWindows()
#press v (exits)
```


# Output Image
![output](https://github.com/vishalbpatil1/yoloface/blob/main/result/result1.png)

The YOLOv3 (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. The published model recognizes 80 different objects in images and videos. For more details, you can refer to this paper.
[Reference link ](https://pjreddie.com/darknet/yolo/)



[Github file source fist](https://github.com/vishalbpatil1/Supper-face-detection-or-crowd-detection)
[Github file source second](https://github.com/vishalbpatil1/yoloface)
