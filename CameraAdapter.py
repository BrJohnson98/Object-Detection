from picamera import PiCamera
from time import sleep

def takePicture(file_name):
    camera = PiCamera()

    sleep(5)
    camera.capture('PiCameraCaptures/' + file_name + '.jpg')
