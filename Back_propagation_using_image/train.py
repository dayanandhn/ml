import os
import numpy as np
import cv2
import random
import pickle


def update():
    print("updating the dataset..")
    path = os.getcwd()
    category = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    Data_Dir = os.path.join(path, 'dataset')
    training = []
    for file in os.listdir(Data_Dir):
        img_dir = os.path.join(Data_Dir, file)
        category.append(file)
        for images in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, images),
                             cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            training.append([img, category.index(file)])
    random.shuffle(training)
    x = []
    y = []
    for features, labels in training:
        x.append(features)
        y.append(labels)
    x = np.array(x).reshape(-1, 150, 150, 1)
    pickle_out = open("training_img.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    pickle_out = open("training_roll_no.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    recognizer.train(x, np.array(y))
    recognizer.save("trainer.yml")
    print("updated..")


update()