import cv2
from face_detect.align_custom import AlignCustom
from face_detect.face_feature import FaceFeature
from face_detect.mtcnn_detect import MTCNNDetect
from face_detect.tf_graph import FaceRecGraph
import sys
import json
import numpy as np
import datetime
import os 
import addface_img as af

'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them
    -> each cropped face is categorized in 3 types: Center, Left, Right
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions.
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)

'''


usb_cam = 0  # select webcam
timestampID = 0
timestampIDlist = {}

cnt = 0
mask_intersection1 = 0
mask_intersection_unknow = 0
dic_mask_intersection = {}
def camera_recog():
    global timestampID,timestampIDlist,timestampID,timestampIDlist,mask_intersection1,mask_intersection2,cnt,dic_mask_intersection

    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    # scale_factor, rescales image for faster detection
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)
    print("[INFO] Inference...")

    vs = cv2.VideoCapture(usb_cam)  # get input from webcam

    while True:
        _, frame = vs.read()
        if cnt == 0:
            sh = frame.shape
            mask_intersection1 = np.ones((sh[0], sh[1])) * 0
            mask_intersection_unknow = np.ones((sh[0], sh[1])) * 0
            cnt = cnt + 1
        # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame, 80)  # min face size is set to 80x80
        if len(rects) == 0:
            timestampID = 0
            timestampIDlist = {}

        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160, frame, landmarks[:, i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print("Align face failed")  # log
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr, positions)


            for (i, rect) in enumerate(rects):
                if (recog_data[i][1]) > 90:
                    x,y,w,h = rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]
                    mask_intersection1 = mask_intersection1 * 0
                    mask_intersection1[y:y+h, x:x+w] = 1
                    dic_mask_intersection[recog_data[i][0]] = mask_intersection1 ##

                    cv2.rectangle(frame, (rect[0], rect[1]),
                                  (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.putText(frame, recog_data[i][0]+" - "+str('%.2f' % recog_data[i][1])+"%", (rect[0],rect[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # print (rect[0],rect[1],rect[2],rect[3])
                else: #Unknow
                    x,y,w,h = rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]
                    mask_intersection_unknow = mask_intersection_unknow * 0
                    mask_intersection_unknow[y:y+h, x:x+w] = 1
                    mask_result = mask_intersection1 * mask_intersection_unknow
                    try:
                        th = cv2.countNonZero(mask_result)/cv2.countNonZero(mask_intersection1)
                    except:
                        th = 0
                    
                    if th < 0.3: #ต่างกันเกิน 30% คิดสะว่าเป็นอีกคน
                        if timestampID == i:
                            timestamp = getTimestamp()
                            timestampID = timestampID + 1
                        try:
                            if timestampIDlist[str(i)][0] >= 0 and timestampIDlist[str(i)][0] <= 30:
                                timestampIDlist[str(i)][0] = timestampIDlist[str(i)][0] + 1
                        except:
                            timestampIDlist[str(i)] = [1,timestamp]
                        
                        x,y,w,h = rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]
                        if timestampIDlist[str(i)][0] % 3 == 0:
                            pathfile = makeDirectoryDataSet(timestampIDlist[str(i)][1])
                            cv2.imwrite(os.path.join(pathfile,timestampIDlist[str(i)][1]+"_"+str(timestampIDlist[str(i)][0])+".jpg"), frame[y:y+h, x:x+w])
                        if timestampIDlist[str(i)][0] == 30:
                            af.add_image_data()
                            moveDirectoryDataSet(timestampIDlist[str(i)][1])
                        # cv2.putText(frame, timestampIDlist[i], (rect[0],rect[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (0, 0, 255), 2)
                        cv2.putText(frame, str(th), (rect[0], rect[1]),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    else:
                        # key = dic_mask_intersection.keys()
                        # for k in key[0]:
                        #     mask_result = dic_mask_intersection[k] * mask_intersection_unknow
                        #     try:
                        #         th = cv2.countNonZero(mask_result)/cv2.countNonZero(dic_mask_intersection[k])
                        #     except:
                        #         th = 0
                        #     if th > 
                        x,y,w,h = rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]
                        mask_intersection1 = mask_intersection1 * 0
                        mask_intersection1[y:y+h, x:x+w] = 1
                        dic_mask_intersection[recog_data[i][0]] = mask_intersection1 ##
                        cv2.rectangle(frame, (rect[0], rect[1]),(rect[2], rect[3]), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)
        cv2.imshow("mask_intersection1", mask_intersection1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''


def findPeople(features_arr, positions, thres=0.6, percent_thres=90):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('face_detect/facerec_128D.txt', 'r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i, features_128D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = "Unknown"
        returnRes.append((result, percentage))
    return returnRes




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


def moveDirectoryDataSet(direct):
    os.rename(os.getcwd() + "/face_detect/dataset/" + direct, os.getcwd() + "/face_detect/dataset2/" + direct)


if __name__ == '__main__':
    camera_recog()
    # os.remove("C:\Users\THAIEAZYELAC\Downloads\FaceRec_Tensorflow\face_detect\dataset\20201008005327105458")
    # os.rmdir("C:\Users\THAIEAZYELAC\Downloads\FaceRec_Tensorflow\face_detect\dataset\20201008005327105458")

    # dir = "20201008005327105458"
  
    # # Path  
    # location = "C:/Users/THAIEAZYELAC/Downloads/FaceRec_Tensorflow/face_detect/dataset/"
    # path = os.path.join(location, dir) 
        
    # # Remove the specified  
    # # file path  
    # os.rmdir(path)  

