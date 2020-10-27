import cv2
from face_detect.mtcnn_detect import MTCNNDetect
from face_detect.tf_graph import FaceRecGraph
import sys
import json
import numpy as np
import datetime
import os 




usb_cam = 0  # select webcam
timestampID = 0
timestampIDlist = {}

def camera_recog():
    global timestampID,timestampIDlist

    MTCNNGraph = FaceRecGraph()
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)
    print("[INFO] Inference...")

    vs = cv2.VideoCapture(usb_cam)  # get input from webcam

    while True:
        _, frame = vs.read()
        # you can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame, 80)  # min face size is set to 80x80
        aligns = []
        positions = []

        print("Face number : ",len(rects))
        if len(rects) == 0:
            timestampID = 0
            timestampIDlist = {}

        for (i, rect) in enumerate(rects):
            if timestampID == i:
                timestamp = getTimestamp()
                # timestampIDlist.append(timestamp)
                timestampID = timestampID + 1
                # print("__________",timestampID,i,timestampIDlist)
            try:
                if timestampIDlist[str(i)][0] >= 0 and timestampIDlist[str(i)][0] < 30:
                    timestampIDlist[str(i)][0] = timestampIDlist[str(i)][0] + 1
                print("^^^^",timestampIDlist)
            except:
                timestampIDlist[str(i)] = [1,timestamp]
                # print("vvvv",timestampIDlist)
            
            pathfile = makeDirectoryDataSet(timestampIDlist[str(i)][1])
            x,y,w,h = rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]
            if timestampIDlist[str(i)][0] % 3 == 0:
                cv2.imwrite(os.path.join(pathfile,timestampIDlist[str(i)][1]+"_"+str(timestampIDlist[str(i)][0])+".jpg"), frame[y:y+h, x:x+w])
            # position = landmarks[:, i]
            # cv2.circle(frame, (position[0], position[5]), 4, (255, 0, 0), -1)
            # cv2.circle(frame, (position[1], position[6]), 4, (255, 0, 0), -1)
            # cv2.circle(frame, (position[2], position[7]), 4, (255, 0, 0), -1)
            # cv2.circle(frame, (position[3], position[8]), 4, (255, 0, 0), -1)
            # cv2.circle(frame, (position[4], position[9]), 4, (255, 0, 0), -1)
            # cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (0, 0, 255), 2)
            # cv2.putText(frame, timestampIDlist[i], (rect[0],rect[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Face Detect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def getTimestamp():
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(':', '')
    timestamp = timestamp.replace('.', '')
    timestamp = timestamp.replace(' ', '')
    return timestamp


def makeDirectoryDataSet(direct):
    path = os.path.join(os.getcwd(),"face_detect\dataset",direct)
    print(path)
    try:  
        os.mkdir(path)  
    except OSError as error:  
        print("Directory is already") 
    return path


if __name__ == '__main__':
    camera_recog()
    
