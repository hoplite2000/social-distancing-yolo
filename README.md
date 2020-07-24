Social Distancing Detection

This repository contains a social-distancing detector built using OpenCV

I have built a classifier whick classifies whether social distancing is followed or violated in real time or in a given video clip. The classifier is built using YOLOV3 pre trained model. I have used OpenCV for real time video capture.

The human_detection directory contains the YOLOV3 model for detecting humans and social_distancing_detector file will detect the humans violating social distancing rule.

To run the social-distancing module execute " python social_distancing_detector.py " in your terminal or command prompt. Press q to stop the program.

REQUIREMENTS:
1. OpenCV
2. imutils
3. Scipy
