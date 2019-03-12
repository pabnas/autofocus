import numpy as np
import cv2

class Camera:
    def __init__(self, camera_file):
        self.ok = False
        self.cap = cv2.VideoCapture(camera_file)
        #self.cap.set(CV_CAP_PROP_SETTINGS, 1);
        self.cap.set(3, 640)  # set the resolution
        self.cap.set(4, 480)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) #turn the autofocus off
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            print("ERROR: File or device not found")
            return
        self.counter = 0
        self.focusTime = 5
        self.correctFocus = 0.5
        self.ok = True

    def __del__(self):
        self.cap.release()

    def get_frame(self, focus):
        if self.ok:
            ret, frame0 = self.cap.read()
            #frame = cv2.resize(frame0,(640,480))
            frame = frame0
            if self.counter%(self.fps*self.focusTime) == 0:
                self.correctFocus = np.random.randint(11)/10.0
            self.counter += 1
            error = abs(focus-self.correctFocus)
            blurOrder = int(error*29.0+1)
            blur = cv2.blur(frame,(blurOrder,blurOrder))
            return blur
        else:
            return None