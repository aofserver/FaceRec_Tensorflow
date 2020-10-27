import cv2
import sys
import json
import time
import numpy as np
import os
from face_detect.tf_graph import FaceRecGraph
from face_detect.align_custom import AlignCustom
from face_detect.face_feature import FaceFeature
from face_detect.mtcnn_detect import MTCNNDetect



'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them
    -> each cropped face is categorized in 3 types: Center, Left, Right
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions.
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)

'''
# imagefile   = 'face_detect/images/example_02.png'
# imagefile   = 'face_detect/images/example_03.jpg'
render_time = 0

######## TEST #######
imagePath = []
path = 'dataset/test'
for fileNames in os.listdir(path):
    imagePath.append(path+"/{}".format(fileNames))
print("imagePaths",imagePath)
#####################



def camera_recog():
    tf_start    = time.time()
    FRGraph     = FaceRecGraph()
    MTCNNGraph  = FaceRecGraph()
    aligner     = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    # scale_factor, rescales image for faster detection
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)

    tf_end      = time.time()
    tf_time     = tf_end - tf_start
    print ("[INFO] Loading time: {:.2f} ms".format(tf_time * 1000))
    #print("[INFO] Inference...")


    for imagefile in imagePath:
        inf_start   = time.time()
        image       = cv2.imread(imagefile)
        rgb         = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(rgb, 80)  # min face size is set to 80x80
        aligns      = []
        positions   = []

        inf_end     = time.time()
        det_time    = inf_end - inf_start
        print ("[INFO] Inference time: {:.2f} ms".format(det_time * 1000))

        rec_start = time.time()

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160, image, landmarks[:, i])
            
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:        ##### Bug
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print("Align face failed")  # log
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data   = findPeople(features_arr, positions)

            for (i, rect) in enumerate(rects):
                if (recog_data[i][1]) > 50:
                    cv2.rectangle(image, (rect[0], rect[1]),
                                    (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.putText(image, recog_data[i][0]+" - "+str('%.2f' % recog_data[i][1])+"%", (rect[0],
                                                                                            rect[1]-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (rect[0], rect[1]),
                                    (rect[2], rect[3]), (0, 0, 255), 2)
                    cv2.putText(image, recog_data[i][0], (rect[0], rect[1]),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                
                pos = landmarks[:, i]
                cv2.circle(image, (pos[0], pos[5]), 4, (255, 0, 0), -1)
                cv2.circle(image, (pos[1], pos[6]), 4, (255, 0, 0), -1)
                cv2.circle(image, (pos[2], pos[7]), 4, (255, 0, 0), -1)
                cv2.circle(image, (pos[3], pos[8]), 4, (255, 0, 0), -1)
                cv2.circle(image, (pos[4], pos[9]), 4, (255, 0, 0), -1)

        cv2.imshow("Face Recognition", image)
        savepath = imagefile.replace("test","output")
        cv2.imwrite(savepath,image)

        rec_end   = time.time()
        rec_time  = rec_end - rec_start
        print ("[INFO] Recognize time: {:.2f} ms".format(rec_time * 1000))
        cv2.waitKey(0)


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


def findPeople(features_arr, positions, thres=0.5, percent_thres=50):
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
    camera_recog()
