# realtime-face-recognition

# 1. Encoding faces
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

After preparing images all you need to do is to run face-encoding.py script. When script stop encoding data, there should be file called "encodedData.pickle".

# 2. Face recognition

If you have successfully encoded your data now you just need to run face-recognition.py script.


API will use main webcam as default. To change it to video saved as for example ".mp4" you need to change line 117 in live_face_recognition.py

To increase performance you can make image smaller by changing argument in getFrame function
