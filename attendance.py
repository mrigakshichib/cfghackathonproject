# import face_recognition
# import cv2
# import numpy as np
# import csv 
# from datetime import datetime

# video_capture = cv2.VideoCapture(0)

# # Load known faces

# meghana_image = face_recognition.load_image_file("Students/Meghana.jpeg")
# meghana_encoding = face_recognition.face_encodings(meghana_image)[0]

# animesh_image = face_recognition.load_image_file("Students/Animesh.jpeg")
# animesh_encoding = face_recognition.face_encodings(animesh_image)[0]

# # varun_image = face_recognition.load_image_file("Students/Varun.jpg")
# # varun_encoding = face_recognition.face_encodings(varun_image)[0]

# known_face_encodings = [  meghana_encoding , animesh_encoding]
# known_face_names = ["Meghana" , "Animesh"]

# # List of expected students

# students = known_face_names.copy()

# face_locations = []
# face_encodings = []

# # Get the current date and time

# now = datetime.now()
# current_date = now.strftime("%Y-%m-%d")

# f = open(f"{current_date}.csv" , "w+" , newline="")
# lnwriter = csv.writer(f)

# while True:
#     _, frame = video_capture.read()
#     small_frame = cv2.resize(frame ,(0,0) , fx=0.25 , fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB) 
    
#     # Recognise faces
#     face_locations = face_recognition.face_locations(rgb_small_frame)
#     face_encodings = face_recognition.face_encodings(rgb_small_frame , face_locations)
    
#     for face_encoding in face_encodings:
#         matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
#         face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
#         best_match_index = np.argmin(face_distance)
        
#         if(matches[best_match_index]):
#             name = known_face_names[best_match_index]
            
#             # Add the text if a person is present
#             if name in known_face_names:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 bottomLeftCornerOfText = (10,100)
#                 fontScale = 1.5
#                 fontColor = (255,0,0)
#                 thickness = 3
#                 lineType = 2
#                 cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                
#                 if name in students:
#                     students.remove(name)
#                     current_time = now.strftime("%H:%M:%S")
#                     lnwriter.writerow([name,current_time])
    
#     cv2.imshow("Camera" , frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
    
# video_capture.release()
# cv2.destroyAllWindows()
# f.close()


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = r"C://Users//dell//Desktop//final//Team-21//static//Students"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         print("pie")

        # nameList = []
        # for line in myDataList:
        #     entry = line.split(',')
        #     nameList.append(entry[0])
        #     if name not in nameList:
        #         now = datetime.now()
        #         dtString = now.strftime('%H:%M:%S')
        #         f.writelines(f'\n{name},{dtString}')

def markAttendance(name):

    with open('Attendance.csv', 'r+') as f:

      
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        print("Writing completed")

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
            print("YAHOOO")

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()