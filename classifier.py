import numpy as np
from cv2 import cv2
from PIL import Image
import os

def train_classifier(data_dir):
    path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    #print(path)
    faces=[]
    ids=[]

    for image in path:
        img=Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        #print(imageNp)
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("trainerr.yml")

train_classifier("data_set")