# Autonomous-Driving-through-Computer-Vision
A suite of Autonomous Driving AI/Computer Vision solutions including traffic-sign detection using RCNN, lane detection, and pedestrian detection made by Jihoon Kim and Martin Liu 

## Traffic Sign Detection (traffic.py, filter.py)
Using a custom made RCNN, we have made a Tensorflow Model trained from the LISA Traffic Sign Dataset.
We then used a variety of filters and contours to detect traffic signs from dash cam footage to feed into the trained model.

## Lane Detection (perspective.py)
We performed lane detection by using various openCV functions. Perspective warping and various filters and blurring were used to make lanes more poignant to our code. A sliding window function kept track of where highest concentration of white pixels (or pixels that defined the lanes) were and line polyfitting gave the coordinates necessary to mark out the lanes.

## Pedestrian Detection (perspective.py)
We implemented an OpenCV model, HOGDescriptor, to detect and box Pedestrians from dash cam footage.
