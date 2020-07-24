from human_detector import config
from human_detector.human_detection import detect_people
from scipy.spatial import distance as dist
import cv2
import imutils
import numpy as np
import os
import sys

#path of the input and output file
inputfilepath = ""
outputfilepath = ""
if len(sys.argv) > 1:
    inputfilepath = sys.argv[1]
    outputfilepath = sys.argv[2]

#load the COCO class labels our YOLO model was trained on
labelspath = os.path.sep.join([config.model_path, "coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

#derive the paths to the YOLO weights and model configuration
weightspath = os.path.sep.join([config.model_path, "yolov3.weights"])
configpath = os.path.sep.join([config.model_path, "yolov3.cfg"])

#load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

#determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(inputfilepath if inputfilepath != "" else 0)
writer = None

#loop over the frames from the video stream
while True:
    (grabbed, frame) = vs.read()

    #if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, index=LABELS.index("person"))

    violate = set()

    #ensure there are *at least* two people detections
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        #loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i,j] < config.min_dist:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        #if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

        #draw the total number of social distancing violations on the output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    if outputfilepath != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputfilepath , fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)
