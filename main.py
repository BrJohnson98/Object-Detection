import CameraAdapter
import YoloAdapter
import time

def main():
    file = 'picture-'+str(time.time())
    CameraAdapter.takePicture(file)
    time.sleep(5)
    YoloAdapter.LocateStudents(file)


if __name__ == "__main__":
    main()
