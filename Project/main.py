import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
from datetime import datetime 

# Inputting the speed limit of the road:
roadLimit=int(input("Enter the speed Limit: "))
 
# Capturing the video file
cap=cv2.VideoCapture("../Videos/cars.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Speed Calculation: 
# Variant-1 (used in the code)
def estimateSpeedViaTime(prevTime,currTime,framesPassed):

    t1 = datetime.strptime(prevTime, "%H:%M:%S.%f")
    t2 = datetime.strptime(currTime, "%H:%M:%S.%f")
    print(t2,t1)
    timeDiff = (t2 - t1).total_seconds()
    print(timeDiff)
    speed=20/timeDiff
    return speed

# Variant-2 (can be used with dynamic ppm)
# with constant ppm it is very suitable for side angle videos
def estimateSpeed(location1, location2,id,fps=int(cap.get(cv2.CAP_PROP_FPS))):
    d_pixels = math.sqrt(math.pow(location2[id][1] - location1[id][1], 2) + math.pow(location2[id][0] - location1[id][0], 2))
    ppm = 8.8
    d_meters = d_pixels/ppm
    speed = d_meters * fps * 3.6
    return speed

# Model used = YOLO (several variants exits nano,large etc)
# Because of hardware constraints we have restricted to nano model 
model=YOLO("../yolov8n.pt")

# Folder storing images of the defaulters
output_folder = 'captured_images for speed more than '+str(roadLimit)
os.makedirs(output_folder, exist_ok=True)

# Classnames as defined in the yolo model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
"boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
"baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
"wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
"broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"]

# Importing a masked image so as to remove filter out unwanted areas in video
mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (740,540),interpolation=cv2.INTER_AREA)

# Tracking (prebuilt tracker from the sort file imported above)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Entry line (x1,y1,x2,y2)
limits = [185, 250, 380, 250]
# Exit line (x1,y1,x2,y2)
limits2=[51,360,380,360]

# for tracking of vehicles
totalCount = []

# These would be used in variant-2 of speed calculation
carLocation1 = {}
carLocation2 = {}

# These used in variant-1 of speed calculation
carEntry={}
carExit={}
speedFound={}

# This is to ensure that a defaulter image is collected only once.
captured=[]
# Another apprach that was tried : counting the number of frames passed
carFrame={}

# Saving the generated video 
output_file = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_file, fourcc, 20.0, (740, 540))


while True:
    suc,frame=cap.read();
    if not suc:
        print("Error: Could not read frame.")
        break

    # resizing frames
    frame=cv2.resize(frame,(740,540),interpolation=cv2.INTER_AREA)

    # The commented 2 lines may be used in the other variants of speed calculation
    # for key in carFrame:
    #     carFrame[key]+=1

    # For visual appearance
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (0, 0))

    # Performing bitwise and on the mask and frame
    imgRegion=cv2.bitwise_and(mask,frame)
    detections = np.empty((0, 5))

    # Supplying the output frame to the yolo model for the analysis.
    results=model(imgRegion,stream=True)
    
    # All the tracked objects are iterated
    for r in results:
        # Every tracked object gives the coordinates of itself being bounded accessed by:
        boxes=r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculating width and height of object using the points
            w, h = x2 - x1, y2 - y1

            # Confidence (bascially how well the object has been detected)
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Capturing Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Performing further operations only when object is vehicle and has atleast 30% confidence.
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                or currentClass == "motorbike" or currentClass=="bicycle" and conf > 0.3:

                # Updating the detections array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # drawing entry line
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    # drawing exit line
    cv2.line(frame, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # calculating widht and height
        w, h = x2 - x1, y2 - y1

        # obtaining center of the object
        cx, cy = x1 + w // 2, y1 + h // 2

        # Variant -2 speed calculation: 
        # if id not in carLocation1:
        #     carLocation1[id]=carLocation2[id]=(cx,cy)
        # else:
        #     carLocation2[id]=carLocation1[id]
        #     carLocation1[id]=(cx,cy)
        # speed=estimateSpeed(carLocation1,carLocation2,id)

        # Bounding boxes for the object
        if id in captured:
            cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(0, 0, 255))
        else:
            cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limits[0]-15 < cx < limits[2]+15 and limits[1] - 22 < cy < limits[3] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                carEntry[id]=str(datetime.now().time());
                carFrame[id]=0;
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if limits2[0]-15 < cx < limits2[2]+15  and (limits2[1]-40) < cy < 500:
            speed=0
            if totalCount.count(id) != 0 and id not in speedFound:
                carExit[id]=str(datetime.now().time())
                speed=estimateSpeedViaTime(carEntry[id],carExit[id],carFrame[id])
                speedFound[id]=speed
                cvzone.putTextRect(frame, f'{speed:.1f} km/hr', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=5)
            elif id in speedFound:
                speed=speedFound[id]
                cvzone.putTextRect(frame, f'{speed:.1f} km/hr', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=5)

            if speed > roadLimit and id not in captured:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(0, 0, 255))
                captured.append(id)
                image_filename = f"captured_{id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                image_path = os.path.join(output_folder, image_filename)
                cv2.imwrite(image_path, frame)
                print(f"Image captured for vehicle {id} with speed {speed} km/hr")

    cv2.putText(frame,str(len(totalCount)),(185,80),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Video",frame) 
    output_video.write(frame)
    
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break

total_set = set(totalCount)
speedFound_set = set(speedFound.keys())
not_in_speed_set = total_set - speedFound_set
not_in_speed_list = list(not_in_speed_set)
print("These id's were captured but their speed wasn't detected: ",end="")
for i in not_in_speed_list:
    print(i,end=" ")
print("\nTotal count: ",len(not_in_speed_list))

output_video.release()
cap.release()
cv2.destroyAllWindows()
