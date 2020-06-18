# realtime-face-recognition 

Tested on Python 3.6 and Ubuntu 18.04.

# 1. face-encoding.py

For encoding faces it's important to have that directory structure in project directory:
  
    project
    ├── cascade-face.xml
    ├── dataset
    │   ├── name1
    │   │   └── example.jpg
    │   ├── name2
    │   │   ├── example.jpg
    │   │   └── example2.jpg
    │   └── name3
    │       └── example.jpg
    ├── face-encoding.py
    ├── face-recognition.py
    └── live_face_recognition.py

On image should be only one person. It doesn't matter if there is only a face or whole body.
Once you have collected all needed images, you need to run "face-encoding.py" script. This script is responsible for encoding images to their 128-d form. After this process you are left with encodedData.pickle file which contains data about faces.

# 2. face-recognition.py

Next step after encoding data is to run main script called "face-recognition.py". It uses video from default webcam (or file) to locate faces on it using OpenCV and cascade classifier. For better performance script uses two separate processes which exchange data between themselves, because of that location of faces is sent to next script which try to recognize a face. When this data comes back to main script, there is an update of names under located faces.

# 3. live_face_recognition.py

This file contains classes used to locate, recognize and display faces.

# 4. Future updates

- Database with last seen date. ✓
