import cv2
import mediapipe as mp
import os
import sqlite3
from PIL import Image

# Khởi tạo đối tượng face_detection
mp_face_detection = mp.solutions.face_detection

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("recognizer/trainingData_2.yml")
recognizer.read("recognizer/trainingData_1.yml")

#get profile by id from DB
def getProfile(id):
    conn = sqlite3.connect('data.db')
    query = "SELECT * FROM people WHERE ID="+ str(id)
    cusror = conn.execute(query)

    profile = None

    for row in cusror:
        profile = row

    return profile

cap = cv2.VideoCapture(0)

fontface = cv2.FONT_HERSHEY_SIMPLEX

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

                roi_gray = frame[y: y+h, x: x+w]
                try:
                    gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                    id, confidence = recognizer.predict(gray)
                    if confidence < 60:
                        profile = getProfile(id)

                        if (profile != None):
                            cv2.putText(frame, ""+str(profile[1]), (x+10, y+h+30), fontface, 1, (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x+10, y+h+30), fontface, 1, (0,0,255), 2)
                except Exception as e:
                    pass


        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
