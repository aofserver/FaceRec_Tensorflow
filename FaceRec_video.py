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
Images from Video Capture -> detect faces' regions -> crop those faces and align them
    -> each cropped face is categorized in 3 types: Center, Left, Right
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions.
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)

'''

# Just disables the AVX2, doesn't enable AVX/FMA
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# videofile = '3person_2.mp4'

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


def camera_recog():
    L_start    = time.time()
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    # scale_factor, rescales image for faster detection
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)
    L_end      = time.time()
    L_time     = L_end - L_start
    print ("[INFO] Loading time: {:.2f} ms".format(L_time * 1000))
    print("[INFO] Video Loading...")

    # vs = cv2.VideoCapture(videofile)  # get input from webcam

    path = r'C:\Users\THAIEAZYELAC\Downloads\FaceRec_Tensorflow\dataset\test\Elon_4.jpg'
    vs = cv2.imread(path)

    while True:

        t1 = cv2.getTickCount()

        # _, frame = vs.read()
        frame = vs.copy()
        tf_start    = time.time()
        # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(
            frame, 80)  # min face size is set to 80x80
        aligns = []
        positions = []

        tf_end      = time.time()
        tf_time     = tf_end - tf_start
        print ("[INFO] Detect time: {:.2f} ms".format(tf_time * 1000))
        rec_start = time.time()
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
                if (recog_data[i][1]) > 80:
                    cv2.rectangle(frame, (rect[0], rect[1]),
                                  (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.putText(frame, recog_data[i][0]+" - "+str('%.2f' % recog_data[i][1])+"%", (rect[0],
                                                                                          rect[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (rect[0], rect[1]),
                                  (rect[2], rect[3]), (0, 0, 255), 2)
                    cv2.putText(frame, recog_data[i][0], (rect[0], rect[1]),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        rec_end   = time.time()
        rec_time  = rec_end - rec_start
        print ("[INFO] Recognize time: {:.2f} ms".format(rec_time * 1000))

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        print ("FPS: {0:.2f}".format(frame_rate_calc))

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA) 
        cv2.imshow("Face Recognition", frame)
        cv2.imwrite("Elon4.jpg",frame)

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


def findPeople(features_arr, positions, thres=0.6, percent_thres=50):
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
