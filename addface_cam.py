import cv2
from face_detect.align_custom import AlignCustom
from face_detect.face_feature import FaceFeature
from face_detect.mtcnn_detect import MTCNNDetect
from face_detect.tf_graph import FaceRecGraph
import sys
import json
import numpy as np
import time

'''
Description:
User input his/her name or ID -> Images from Video Capture -> detect the face -> crop the face and align it 
    -> face is then categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Append each newly extracted face 128D vector to its corresponding position type (Center, Left, Right)
    -> Press Q to stop capturing
    -> Find the center ( the mean) of those 128D vectors in each category. ( np.mean(...) )
    -> Save
    
'''

usb_cam = 0  # select webcam
MTCNNGraph = FaceRecGraph()
aligner = AlignCustom()
extract_feature = FaceFeature(FRGraph)
face_rec = MTCNNDetect(MTCNNGraph, scale_factor=2)

def create_manual_data():
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_rec = MTCNNDetect(MTCNNGraph, scale_factor=2)

    vs = cv2.VideoCapture(usb_cam)  # get input from webcam
    start = time.time()
    PERIOD_OF_TIME = 15  # input time to break
    print('Input your name..')
    new_name = input()  # get in add_user
    f = open('face_detect/facerec_128D.txt', 'r')
    data_set = json.loads(f.read())
    person_imgs = {"Left": [], "Right": [], "Center": []}
    person_features = {"Left": [], "Right": [], "Center": []}
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset")
    while True:
        _, frame = vs.read()
        rects, landmarks = face_rec.detect_face(
            frame, 80)  # min face size is set to 80x80
        aligns = []
        positions = []
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160, frame, landmarks[:, i])
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                aligns.append(aligned_frame)
                positions.append(pos)

        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr, positions)
            for (i, rect) in enumerate(rects):
                # draw bounding box for the face
                cv2.rectangle(frame, (rect[0], rect[1]),
                              (rect[2], rect[3]), (0, 255, 0), 2)

        cv2.imshow("Captured face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # break function in time
        if time.time() > start + PERIOD_OF_TIME:
            break

    for pos in person_imgs:  # there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [
            np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]
    data_set[new_name] = person_features
    f = open('face_detect/facerec_128D.txt', 'w')
    f.write(json.dumps(data_set))

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


if __name__ == '__main__':
    create_manual_data()
