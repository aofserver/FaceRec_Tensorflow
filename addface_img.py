#https://github.com/vudung45/FaceRec

import cv2
import os
from face_detect.align_custom import AlignCustom
from face_detect.face_feature import FaceFeature
from face_detect.mtcnn_detect import MTCNNDetect
from face_detect.tf_graph import FaceRecGraph
from imutils import paths

import json
import numpy as np
import time

# Just disables the AVX2, doesn't enable AVX/FMA
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# FRGraph    = 0
# MTCNNGraph = 0
# aligner    = 0
# extract_feature = 0
# face_rec = 0

# def initfn():
#     global FRGraph,MTCNNGraph,aligner,extract_feature,face_rec
#     FRGraph    = FaceRecGraph()
#     MTCNNGraph = FaceRecGraph()
#     aligner    = AlignCustom()
#     extract_feature = FaceFeature(FRGraph)
#     face_rec   = MTCNNDetect(MTCNNGraph, scale_factor=2)
#     print("))))))))))))))))))))))))))))))))))))))))))")


FRGraph    = FaceRecGraph()
MTCNNGraph = FaceRecGraph()
aligner    = AlignCustom()
extract_feature = FaceFeature(FRGraph)
face_rec   = MTCNNDetect(MTCNNGraph, scale_factor=2)


def add_image_data():
    global FRGraph,MTCNNGraph,aligner,extract_feature,face_rec
    
    for folderNames in os.listdir('face_detect/dataset'):
        imagePaths = list(paths.list_images(
            "face_detect/dataset/%s" % folderNames))
        f = open('face_detect/facerec_128D.txt', 'r')
        data_set = json.loads(f.read())
        person_imgs     = {"Left": [], "Right": [], "Center": []}
        person_features = {"Left": [], "Right": [], "Center": []}


        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
            image = cv2.imread(imagePath)
            rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rects, landmarks = face_rec.detect_face(rgb, 80)  # min face size is set to 80x80
            
            for (i, rect) in enumerate(rects):
                aligned_frame, pos = aligner.align(160, rgb, landmarks[:, i])
                if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                    person_imgs[pos].append(aligned_frame)

        for pos in person_imgs:  # there r some exceptions here, but I'll just leave it as this to keep it simple
            person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]

        data_set[folderNames] = person_features
        f = open('face_detect/facerec_128D.txt', 'w')
        f.write(json.dumps(data_set))
    print("Add Image Data success")


# if __name__ == '__main__':
#     add_image_data()
