import cv2
import mediapipe as mp
import os
import numpy as np
import sqlite3

def insertOrUpdate(id, name):
    conn = sqlite3.connect('data.db')

    query = "SELECT * FROM people WHERE ID="+ str(id)
    cusror = conn.execute(query)

    isRecordExist = 0
    for row in cusror:
        isRecordExist = 1

    if(isRecordExist == 0):
        query = "INSERT INTO people(ID, Name) VALUES("+str(id)+ ",'"+str(name)+ "')"
    else:
        query = "UPDATE people SET Name='"+str(name)+"' WHERE ID="+str(id)
    conn.execute(query)
    conn.commit()
    conn.close()

# Khởi tạo đối tượng face_detection
mp_face_detection = mp.solutions.face_detection

#load lib
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

#insert to db
id = input("Enter your ID:")
name = input("Enter your Name: ")
insertOrUpdate(id, name)

sampleNum = 0

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Tạo thư mục riêng cho mỗi người
                folder_path = os.path.join('dataSet', str(id))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                sampleNum += 1

                cv2.imwrite(os.path.join(folder_path, str(name)+'_'+str(id)+'_'+str(sampleNum)+ '.jpg'), frame[y: y + h, x: x+w])

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        if sampleNum > 50:
            break

cap.release()
cv2.destroyAllWindows()
