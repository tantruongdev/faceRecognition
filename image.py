
import cloudinary

# Import the cloudinary.api for managing assets
import cloudinary.api

# Import the cloudinary.uploader for uploading assets
import cloudinary.uploader

# Import os to get the environment variables
import os
import shutil
import requests

cloudinary.config(
    cloud_name="dkjd1omoz",
    api_key="668941927865965",
    api_secret="Ne6WXHNT0T-3ia52vnKiIjd6It8",
    secure=True,
)


# Hàm để xóa thư mục và tất cả ảnh trong thư mục đó
def delete_folder_cloudinary(folder_name):
    # Lấy danh sách tất cả tệp tin trong thư mục
    resources = cloudinary.api.resources(
        type="upload",
        prefix=folder_name,  # Tên thư mục cần xóa
        max_results=1000,  # Số lượng tệp tin tối đa cần lấy, tăng nếu cần thiết
    )["resources"]

    # Lặp qua từng tệp tin và xóa chúng
    for resource in resources:
        public_id = resource["public_id"]
        result = cloudinary.uploader.destroy(public_id)
        print(f"Đã xóa ảnh: {public_id} - Kết quả: {result['result']}")

    result = cloudinary.api.delete_folder(folder_name)
    print(f"Đã xóa thư mục: {folder_name}")


def deleteUser(username):
    delete_folder_cloudinary("train/" + username)
    delete_folder_cloudinary("users/" + username)
    if os.path.exists("knn_folder/train/" + username):
        shutil.rmtree("knn_folder/train/" + username)
    if os.path.exists("knn_folder/train/" + username):
        os.rmdir("knn_folder/train/" + username)



def downloadUser(username, folder_name_user):
    # Tên của thư mục bạn muốn tải về
    folder_name = folder_name_user + username

    # Lấy danh sách các tệp trong thư mục
    result = cloudinary.api.resources(type="upload", prefix=folder_name)
    # Lặp qua danh sách các tệp và tải về từng tệp một
    for resource in result["resources"]:
        print(f'Đang tải xuống tệp {resource["public_id"]}.{resource["format"]}')
        url = (
            "https://res.cloudinary.com/dkjd1omoz/image/upload/v1/"
            + resource["public_id"]
            + "."
            + resource["format"]
        )
        print(url)
        local_folder = "knn_folder/train/" + username
        file_name = resource["public_id"].split("/")[-1]
        local_filename = file_name + "." + resource["format"]
        path = os.path.join(local_folder, local_filename)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        r = requests.get(url)
        # save the image received into the file
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Tệp {local_filename} đã được tải xuống")

    print("Tất cả các tệp đã được tải xuống")
