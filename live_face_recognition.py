import math
from multiprocessing import Process, Queue
import random
import face_recognition
import cv2
import pickle

recognitionQueue = Queue()
displayQueue = Queue()


class FaceTracking(object):

    def _getDifference(self, faces, ids):
        index_list = []
        for face in faces:
            new = True
            for index, id in enumerate(ids):
                diff = 0
                for x1, x2 in zip(face, id[1]):
                    diff += math.sqrt(pow(x1 - x2, 2))

                if diff <= 70:
                    ids[index][1] = face
                    index_list.append(index)
                    new = False
                    break

            if new:
                rand = random.randrange(0, 100)
                ids.append([rand, face])
                index_list.append(len(ids) - 1)

        return index_list, ids

    def _deleteOld(self, index_list, ids):
        for x in range(len(ids)):
            if x not in index_list:
                try:
                    del ids[x]
                except:
                    pass

        return ids

    def track(self, faces, ids):
        self.index_list, self.ids = self._getDifference(faces, ids)
        self.ids = self._deleteOld(self.index_list, self.ids)

        return self.ids


class FaceRecognition:

    def __init__(self, data):
        self.encoded = data

    def _getQueuedItems(self):
        image, ids = recognitionQueue.get()
        return image, ids

    def _encodeFaces(self, image, ids):
        faces = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for object in ids:
            faces.append(object[1])

        encodings = face_recognition.face_encodings(rgb, faces)

        return encodings

    def _getMatches(self, encodings):
        namesTmp = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.encoded["encodings"],
                                                     encoding)
            faceName = "Unknown"

            if True in matches:
                count = {}
                matched = [i for (i, b) in enumerate(matches) if b]

                for i in matched:
                    faceName = self.encoded["names"][i]
                    count[faceName] = count.get(faceName, 0) + 1

                faceName = max(count, key=count.get)

            namesTmp.append(faceName)

        return namesTmp

    def _prepareIds(self, ids, names):
        readyIds = []

        for id, name in zip(ids, names):
            readyIds.append([id[0], name])

        return readyIds

    def recognizeFace(self):
        while True:
            image, ids = self._getQueuedItems()
            encodings = self._encodeFaces(image, ids)
            matchedIds = self._getMatches(encodings)
            readyIds = self._prepareIds(ids, matchedIds)

            displayQueue.put(readyIds)


class Display(object):

    def __init__(self, data):
        self.faceRecognition = FaceRecognition(data)
        self.faceTracking = FaceTracking()
        self.cap = cv2.VideoCapture(0)
        self.faceCascade = cv2.CascadeClassifier("cascade-face.xml")
        self.ids = [[0, []]]

    def getFrame(self, scale):
        ret, image = self.cap.read()
        self.oryginalImage = image.copy()
        self.scale = scale
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dstSize = (width, height)
        image = cv2.resize(image, dstSize)

        return image

    def locateFaces(self, image):
        rawFaces = self.faceCascade.detectMultiScale(image,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(30, 30))

        faces = []
        if len(rawFaces) != 0:
            for face in rawFaces:
                x, y, w, h = face
                faces.append((y, x + w, y + h, x))

        return faces

    def startUp(self, image, ids):
        faceRecognition_proc = Process(target=self.faceRecognition.recognizeFace)
        faceRecognition_proc.start()
        recognitionQueue.put((image, ids))

    def exchangeData(self, image, ids, names):
        try:
            recognizedNames = displayQueue.get_nowait()
            recognitionQueue.put((image, ids))
            return recognizedNames
        except:
            return names

    def matchNamesToIDs(self, ids, recognized):
        idDictionary = {}
        matchedNames = []
        self.recognizedNames = recognized

        for object in ids:
            idDictionary[object[0]] = object[1]

        if len(idDictionary) == 0:
            self.recognizedNames = []

        for name in self.recognizedNames:
            if name[0] in idDictionary:
                matchedNames.append((name[1], idDictionary[name[0]]))

        return matchedNames

    def updateFrame(self, matched, fullscreen):

        self.matchedNames = matched
        image = self.oryginalImage
        self.originalScale = 1 / self.scale

        for (name, (top, right, bottom, left)) in self.matchedNames:
            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(image,
                          (int(left * self.originalScale), int(top * self.originalScale)),
                          (int(right * self.originalScale), int(bottom * self.originalScale)),
                          color, 2)
            cv2.putText(image, str(name), (int(left * self.originalScale), int(top * self.originalScale) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)

        if fullscreen:
            cv2.namedWindow("Result", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("Result", image)
