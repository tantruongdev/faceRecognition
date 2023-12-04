
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
uri = "mongodb+srv://thaole04:041003@facerecognition.tcpw49j.mongodb.net/?retryWrites=true&w=majority"
def connectMongoDB():
    # Kết nối tới MongoDB
    try:
        client = MongoClient(uri)
        db = client['mydatabase']
        print("Kết nối thành công!")
        return db
    except ConnectionFailure:
        print("Không thể kết nối tới MongoDB. Vui lòng kiểm tra kết nối.")
def getControl():
    db = connectMongoDB()
    # get data from collection controls
    controls = db["controls"]
    # convert to list
    list_controls = list(controls.find({"executed": False}))
    return list_controls

def updateControl(id):
    db = connectMongoDB()
    controls = db["controls"]
    controls.update_one({"_id": id}, {"$set": {"executed": True}})
def failControl(id):
    db = connectMongoDB()
    controls = db["controls"]
    controls.update_one({"_id": id}, {"$set": {"executed": False}})
if __name__ == "__main__":
    connectMongoDB()
    controls = getControl()
    for control in controls:
        print(control)
