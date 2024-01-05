# Real-time Object Detection and Distance Estimation using YOLOv8
This Python script utilizes YOLOv8 (YOLOv8s.pt) from Ultralytics for real-time object detection and calculates the distance of detected objects from the camera in inches.

## Prerequisites
- OpenCV (`cv2`)
- Ultralytics (`YOLO`)

## Setup
1. OpenCV: To install --> ```pip install opencv-python```
2. Ultralytics: To install --> ```pip install ultralytics```
3. torch (Optional): To run the code on GPU you have to have pytorch and Nvidia CUDA toolkit. You have to check the latest version of pytorch in https://pytorch.org/ in my case it is 12.1.0
![image](https://github.com/SpawnedNPC/DistanceEstimationYoloV8/assets/125773427/d08d34d1-3d93-4f69-bf82-535cbac9efde) and based on this we have to install same version of CUDA from this website:
 https://developer.nvidia.com/cuda-toolkit-archive. And run this code: ``` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ```

## 
