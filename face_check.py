import cv2
import dlib
import numpy as np

# Đường dẫn tới file pre-trained model dlib
dlib_model_path = "path_to_dlib_model.dat"

# Đường dẫn và tên file ảnh mẫu
image_paths = [
    "pic/elon check.jpg",
    "pic/elon musk.jpg",
    "pic/tokuda.jpg",
    "pic/truong tan.jpg"
]

# Khởi tạo bộ nhận diện khuôn mặt dlib
detector = dlib.get_frontal_face_detector()

# Khởi tạo bộ nhận dạng khuôn mặt dlib
facerec = dlib.face_recognition_model_v1(dlib_model_path)

# Khởi tạo danh sách các biểu diễn khuôn mặt mẫu
known_face_descriptors = []

# Tải và tính toán biểu diễn khuôn mặt mẫu từ ảnh
for image_path in image_paths:
    # Đọc ảnh mẫu
    img = cv2.imread(image_path)
    # Chuyển đổi ảnh mẫu sang đen trắng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ nhận diện khuôn mặt để xác định khuôn mặt trong ảnh
    faces = detector(gray)

    # Lấy biểu diễn khuôn mặt từ ảnh mẫu
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        known_face_descriptors.append(face_descriptor)

# Sử dụng camera để chụp hình ảnh
camera = cv2.VideoCapture(0)

while True:
    # Đọc hình ảnh từ camera
    ret, frame = camera.read()

    # Chuyển đổi hình ảnh sang đen trắng để tăng tốc độ nhận diện
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ nhận diện khuôn mặt để xác định các khuôn mặt trong hình ảnh
    faces = detector(gray)

    # Duyệt qua các khuôn mặt và vẽ hộp giới hạn xung quanh chúng
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)

        # So sánh biểu diễn khuôn mặt với các biểu diễn mẫu
        distances = np.linalg.norm(known_face_descriptors - face_descriptor, axis=1)
        min_distance = np.min(distances)

        # Xác định người có khoảng cách nhỏ nhất
        if min_distance < 0.6:
            min_distance_index = np.argmin(distances)
            recognized_person = f"Person {min_distance_index + 1}"
        else:
            recognized_person = "Unknown"

        # Vẽ hộp giới hạn và tên người được nhận diện
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị hình ảnh từ camera với khuôn mặt đã được nhận diện
    cv2.imshow("Face Recognition", frame)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
camera.release()
cv2.destroyAllWindows()