"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in a live stream by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in a live video stream.

NOTE: This example requires scikit-learn, opencv and numpy to be installed! You can install it with pip:

$ pip3 install scikit-learn
$ pip3 install numpy
$ pip3 install opencv-contrib-python

"""

import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
from gpiozero import LED, Button, Buzzer
import time
from server import controlUnit


# for cloudinary
import cloudinary
# Import the cloudinary.api for managing assets
import cloudinary.api
# Import the cloudinary.uploader for uploading assets
import cloudinary.uploader
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
cloudinary.config(
    cloud_name="dkjd1omoz",
    api_key="668941927865965",
    api_secret="Ne6WXHNT0T-3ia52vnKiIjd6It8",
    secure=True,
)

# for database
from pymongo import MongoClient
uri = "mongodb+srv://thaole04:041003@facerecognition.tcpw49j.mongodb.net/?retryWrites=true&w=majority"

# for control gpio
SW1 = Button(21)
SW2 = Button(16)
SW3 = Button(20)
BUZZER = Buzzer(26)
LED = LED(17)


# def active buzzer and LED
def notice():
    BUZZER.on()
    LED.on()
    time.sleep(0.2)
    BUZZER.off()
    LED.off()

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors = 1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

# Show predicttion label for display in monitor
def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        # text_width, text_height = draw.textsize(name)
        text_height = 30 
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":
    print('Press button following this manual:')
    print('1. Press button 1 to active face recognition function.')
    print('2. Press button 2 to update face recognition system.')
    print('3. Press button 3 to exit system.')
    while True:
        if SW1.is_pressed:
            # check for trained model 
            if not os.path.isfile("trained_knn_model.clf"):
                print("Training KNN classifier...")
                classifier = train("knn_folder/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
                print("Load complete!")
            else:
                print("Trained model founded, loading...")
                classifier = pickle.load(open("trained_knn_model.clf", 'rb'))
                print("Load complete!")
            # process one frame in every 30 frames for speed
            process_this_frame = 29
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if ret:
                    # Different resizing options can be chosen based on desired program runtime.
                    # Image resizing for more stable streaming
                    img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    process_this_frame = process_this_frame + 1
                    if process_this_frame % 30 == 0:
                        predictions = predict(img, model_path="trained_knn_model.clf")
                        if (predictions != []):
                            print(predictions)
                            try:
                                # save img to folder knn_folder/activities with name is HH_MM_SS_DD_MM_YY.jpg
                                cv2.imwrite("knn_folder/activities/"+time.strftime("%H_%M_%S_%d_%m_%Y")+".jpg", frame)
                                # save img to cloudinary
                                response = cloudinary.uploader.upload(
                                    "knn_folder/activities/"+time.strftime("%H_%M_%S_%d_%m_%Y")+".jpg", 
                                    folder = "activities",
                                    public_id = time.strftime("%H_%M_%S_%d_%m_%Y"),
                                )
                                secured = False
                                for prediction in predictions:
                                    if prediction[0] != "unknown":
                                        secured = True
                                        break;
                                for prediction in predictions:
                                    # create data to insert in mongodb
                                    data = {
                                        "time": time.strftime("%H-%M-%S"),
                                        "date": time.strftime("%d-%m-%Y"),
                                        "username": prediction[0],
                                        "image": response['secure_url'],
                                        "secured": secured
                                    }
                                    # connect to mongodb
                                    client = MongoClient(uri)
                                    db = client['mydatabase']
                                    activities = db["activities"]
                                    # insert data to collection activities
                                    activities.insert_one(data)
                                    print("Success!")
                                notice()
                                cap.release()
                                break;
                            except:
                                print("Error!")
                                notice()
                                cap.release()
                                break;
        elif SW2.is_pressed:
            try:
                controlUnit()
                # retrain model_path
                print("Training KNN classifier...")
                try:
                    classifier = train("knn_folder/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
                    print("Training complete!")
                except:
                    print("Training failed!")
            except:
                print("Error!")
        elif SW3.is_pressed:
            print("Exit system!")
            exit(0)
        else:
            pass
