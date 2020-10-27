import cv2
from face_detect.mtcnn_detect import MTCNNDetect
from face_detect.tf_graph import FaceRecGraph
import sys
import json
import numpy as np


usb_cam = 0  # select webcam

def camera_recog():
    MTCNNGraph = FaceRecGraph()
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)
    print("[INFO] Inference...")

    vs = cv2.VideoCapture(usb_cam)  # get input from webcam

    while True:
        _, frame = vs.read()
        # you can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame, 80)  # min face size is set to 80x80
        print("__________________len(rects)  ",len(rects))
        print("__________________landmarks  ",landmarks)
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            position = landmarks[:, i]
            cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (0, 255, 0), 2)
            cv2.circle(frame, (position[0], position[5]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (position[1], position[6]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (position[2], position[7]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (position[3], position[8]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (position[4], position[9]), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow("Face Detect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()



if __name__ == '__main__':
    camera_recog()
