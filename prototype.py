import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import sys

path = 'E:\\face attendance\\studentimages'

images = []
classNames = []

mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)

def markAttendance(name):
    attendance_path = 'E:\\face attendance\\Attendance.csv'
    
    # Get the current date and time
    now = datetime.now()
    date = now.strftime('%d-%B-%Y')  # Correct date format
    time = now.strftime('%I:%M:%S:%p')
    
    # Read existing data
    with open(attendance_path, 'w+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList if date in entry]
        
        # Prepare the attendance entry
        attendance_entry = f'{name},{time},{date}\n'

        # Write attendance if not already present for today
        if name not in nameList:
            f.write(attendance_entry)
            print(f'{name} marked present at {time} on {date}')
        else:
            print(f'{name} is already marked present today.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)

    key = cv2.waitKeyEx(1)
    if key == ord('q'):
        break

# Release the video capture
cap.release()

def stop_camera():
    cap.release()  # Release the camera
    root.destroy()  # Close the Tkinter window
    cv2.destroyAllWindows()  # Close any OpenCV windows
    sys.exit(0)

root = tk.Tk()
root.title("Face Recognition Attendance System")



# Display a video feed in the GUI
video_label = tk.Label(root)
video_label.pack()

def update_video():
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    video_label.img = img
    video_label.config(image=img)
    video_label.after(10, update_video)

update_video()
root.mainloop()