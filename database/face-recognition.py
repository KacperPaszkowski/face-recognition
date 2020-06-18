import live_face_recognition as lfr
import cv2
import pickle

ids = [[0, []]]
firstTime = True

data = pickle.loads(open("encodedData.pickle", "rb").read())

faceTracking = lfr.FaceTracking()
faceRecognition = lfr.FaceRecognition(data)
display = lfr.Display(data)
database = lfr.Database()
database.makeDatabase()


matchedNames = []
names = []


while True:
    image = display.getFrame(0.7)
    faces = display.locateFaces(image)

    ids = faceTracking.track(faces, ids)

    if firstTime and __name__ == "__main__":
        display.startUp(image, ids)
        firstTime = False

    names = display.exchangeData(image, ids, names)
    matchedNames = display.matchNamesToIDs(ids, names)
    database.checkIfGone(matchedNames)
    databaseContent = database.getDatabase()
    display.updateFrame(matchedNames, databaseContent, False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
