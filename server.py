
from mongodb import getControl, updateControl, failControl
from image import downloadUser, deleteUser

def controlUnit():
    list_controls = getControl()
    for control in list_controls:
        try:
            if control['type'] == 'AddUser':
                downloadUser(control['username'], "train/")
                downloadUser(control['username'], "users/")
                updateControl(control['_id'])
            elif control['type'] == 'DeleteUser':
                deleteUser(control['username'])
                updateControl(control['_id'])
            elif control['type'] == 'AddPhotos':
                downloadUser(control['username'], "train/")
                downloadUser(control['username'], "users/")
                updateControl(control['_id'])
            else:
                failControl(control['_id'])
        except:
            print("Error while update system!")
            break
