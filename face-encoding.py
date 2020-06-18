import pickle
import face_recognition
import cv2
import os

encoded = []
encodedNames = []

dataset_path = "".join(os.path.split(os.path.realpath(__file__))[:-1]) + "/dataset/"
names = os.listdir(dataset_path)

for name in names:
    images = os.listdir(dataset_path + name)

    for img in images:
        image = cv2.imread(dataset_path + name + "/" + img)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("locating {}...".format(name))
        face = face_recognition.face_locations(rgb)
        print("encoding...")
        encodings = face_recognition.face_encodings(rgb, face)
        for encoding in encodings:
            encoded.append(encoding)
            encodedNames.append(name)

data = {"encodings": encoded, "names": encodedNames}

file = open("encodedData.pickle", "wb")
file.write(pickle.dumps(data))
file.close()
