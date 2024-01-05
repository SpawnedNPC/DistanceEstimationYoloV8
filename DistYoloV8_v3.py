import cv2
from ultralytics import YOLO
#import torch

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

#torch.cuda.set_device(0) # Uncommenting this line and line 3 if you want to run the code on GPU(if available)

# Constants
KNOWN_DISTANCE = 45.0  # Inches
CONFIDENCE_THRESHOLD = 0.4
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Camera setup
camera = cv2.VideoCapture(0) #Specify the index of the camera which you want work with (if multiple camera is connected to the system)
camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define class names and their real-world widths (in inches)
class_widths = {
    "person": 20.0,
    "cell phone": 3.5,
    #"car": 70.0,
    # If you want to include more classes, add class here along with the real life width of that class
}

class_names = model.names # this will give the dictinory of classes that are used in this model with their class_ids as keys

# Function to calculate focal length
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Function to calculate distance
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return round(distance, 2)

# Calculate focal lengths for each class
focal_lengths = {}
for class_name, real_width in class_widths.items():
    ref_image = cv2.imread(f"reference_images/{class_name}.jpg")  # Change the path accordingly
    ref_image = cv2.resize(ref_image, (int(camera_width), int(camera_height)))
    
    result = model(source=ref_image, conf=CONFIDENCE_THRESHOLD)
    for i,j in class_names.items():
        if i in result[0].boxes.cls.tolist():
            #print(j)
            if j == class_name:
                box=result[0].boxes.xyxy[result[0].boxes.cls.tolist().index(i)].tolist()
                class_width_in_rf = box[2]-box[0]
                
        
    focal_lengths[class_name] = focal_length_finder(KNOWN_DISTANCE, real_width, class_width_in_rf)

# Real-time object detection and distance estimation
while True:
    success, frame = camera.read()

    if not success:
        print("Error reading camera frame!")
        break

    results = model(source=frame, conf=CONFIDENCE_THRESHOLD)
    boxes = results[0].boxes.xyxy.tolist()
    class_ids=results[0].boxes.cls.tolist()

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box

        width_in_frame = x2 - x1
        
        #Here I am calculating the distance of the objects that we mentioned in class_widths dictionary or else it will give it as Unknown distance.
        class_name = class_names[int(class_id)]
        if class_name in class_widths and class_name in focal_lengths:
            distance = distance_finder(focal_lengths[class_name], class_widths[class_name], width_in_frame)
        else:
            distance = "Unknown"

        #Drawing the rectangles (Bounding boxes) based on the detection and giving the text to respective Bboxes
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[int(class_id) % len(COLORS)], 2)
        if not isinstance(distance, (int, float)):
            distance = str(distance)
        cv2.putText(frame, f"{class_name}: {distance} Inches", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS[int(class_id) % len(COLORS)], 2)

    cv2.imshow("YOLOv8 Camera Detections", frame)
    
    
# Press q to close the camera and end the code.
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
