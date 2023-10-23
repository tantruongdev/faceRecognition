import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Tạo đường dẫn tới thư mục chứa ảnh
path = 'dataSet'

def getImageWithId(path, id):
    imagePaths = [os.path.join(path, str(id), f) for f in os.listdir(os.path.join(path, str(id)))]

    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')

        faceNp = np.array(faceImg, 'uint8')

        # Extract ID from file name (assuming the format is name_id_sampleNum.jpg)
        Id = int(os.path.basename(imagePath).split('_')[1])

        faces.append(faceNp)
        IDs.append(Id)

        cv2.imshow('training', faceNp)
        cv2.waitKey(10)

    return faces, IDs


# Lấy danh sách khuôn mặt và IDs từ thư mục
for id in os.listdir(path):
    faces, Ids = getImageWithId(path, id)

    recognizer.train(faces, np.array(Ids))

    if not os.path.exists('recognizer'):
        os.makedirs('recognizer')
    recognizer.save(f'recognizer/trainingData_{id}.yml')

cv2.destroyAllWindows()
