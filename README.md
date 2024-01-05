# Real-time Object Detection and Distance Estimation using YOLOv8
This Python script utilizes YOLOv8 (YOLOv8s.pt) from Ultralytics for real-time object detection and calculates the distance of detected objects from the camera in inches.

## Prerequisites:
- OpenCV (`cv2`)
- Ultralytics (`YOLO`)

## Setup:
1. OpenCV: To install --> ```pip install opencv-python```
2. Ultralytics: To install --> ```pip install ultralytics```
3. torch (Optional): To run the code on GPU you have to have pytorch and Nvidia CUDA toolkit. You have to check the latest version of pytorch in https://pytorch.org/ in my case it is 12.1.0
![image](https://github.com/SpawnedNPC/DistanceEstimationYoloV8/assets/125773427/d08d34d1-3d93-4f69-bf82-535cbac9efde) and based on this we have to install same version of CUDA from this website:
 https://developer.nvidia.com/cuda-toolkit-archive. And run this code: ``` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ```

## DistYoloV8_v3.py:
This code is used to detect and estimate the distance of an object from the camera which runs on the pre trained model of YOLOV8 (yolov8s.pt) that was trained on COCO dataset.
Here I am calculating distance of the person and cell phone from camera in inchs. We can also calculate the distance of the other objects simply by mentioning the name of the object and real life width of the object in inchs in the ```class_widths``` dictionary.

For example, lets say you want to estimate the distance of water bottle from the camera. Put ```bottle: 2.5``` in the class_width

```
# Define class names and their real-world widths (in inches)
class_widths = {
    "person": 20.0,
    "cell phone": 3.5,
    #"bottle": 2.5,
    # If you want to include more classes, add class here along with the real life width of that class
}
```

## reference image folder:
In this folder we have the images of each object for which we are going to estimate the distance. For example, this is how the cell phone.jpg will be:
![image](https://github.com/SpawnedNPC/DistanceEstimationYoloV8/assets/125773427/42985c42-cdb6-4ae7-bf0d-d78e855b5d51)

We have to upload the images of objects which are captured at a known distance. In may case I have took the picture of my cell phone at 45 inchs from my laptop camera.

Note: There is a person and cell phone in cell phone reference image. Model might detect the both classes. But dont worry, even if the image have multiple classes in reference image of a particular class, code was written in such a way that it only considers the class it should refers to.

For example, lets say you want to estimate the distance of water bottle from the camera. Put ```bottle: 2.5``` in the class_width where 2.5 is width of the bottle in inchs in the reality. And Upload the image of the bottle taken at known distance (say 45 inchs from the camera).


## How it works
- The script first loads the YOLOv8 model (`yolov8s.pt`) for object detection.
- Variables such as known distance, and confidence threshold can be modified based on you requirement.
- Camera setup is initialized using OpenCV (`cv2.VideoCapture`).
- Class names and their real-world widths in inches are defined.
- Functions are provided to calculate focal length and distance based on object width in the frame.
- Focal lengths for each class are calculated using reference images.
- Real-time object detection and distance estimation are performed in a continuous loop.
- Detected objects' distances are displayed on the frame in real-time.
- Press 'q' to quit the application.

## Important Note:
- A critical consideration when utilizing the camera to estimate an object's distance is that the calculation of distance relies solely on the object's width as it appears within the camera frame. The height of the object, in this context, doesn’t play a role in determining the distance. This implies that when the camera is pointed at an object for distance estimation, what matters is the horizontal span of the object that appears within the frame. For accurate distance calculations, the script evaluates the width of detected objects based on their appearance in the captured video or image feed.
  
- You can modify the units of the distance by simply applying the formulas or capture the object images in required units.
- If there are more than one camera connected to PC, then modify the index here accordingly ```camera = cv2.VideoCapture(0```.
  

