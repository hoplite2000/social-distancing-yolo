from .config import min_conf
from .config import nms_thresh
import cv2
import numpy as np

def detect_people(frame,net,ln,index=0):
    (H,W) = frame.shape[:2]
    results = []

    #create a blob of the image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutput = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    #loop over each of the layer outputs
    for output in layerOutput:
        #loop over each of the detections
        for detection in output:
            scores = detection[5:]
            ID = np.argmax(scores)
            confidence = scores[ID]

            if ID == index and confidence > min_conf:
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #apply non-max supression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

    if len(idxs)>0:
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results
