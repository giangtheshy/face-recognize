import datetime
import numpy as np
import cv2
from imutils.video import FPS

class Detector:
    def __init__(self):
        self.faceModel = cv2.dnn.readNetFromCaffe('model/opencv_face_detection/res10_300x300_ssd_iter_140000.prototxt',caffeModel="model/opencv_face_detection/res10_300x300_ssd_iter_140000.caffemodel")

    def processImage(self ,imageName):
        self.img = cv2.imread(imageName)
        (self.height, self.width) = self.img.shape[:2]

        self.processFrame()

        cv2.imshow("Output", self.img)
        cv2.waitKey(0)

    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)

        if cap.isOpened() == False:
            print("Error opening video stream or file")
            return
        
        (success, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]

        fps = FPS().start()

        while success:
            self.processFrame()
            cv2.imshow("Output", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
            success, self.img = cap.read()
        
        fps.stop() 
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        cap.release()
        cv2.destroyAllWindows()


    def processFrame(self):
        time_start = datetime.datetime.now().timestamp()
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123.0),swapRB = False,crop = False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.3:
                box = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin,ymin,xmax,ymax) = box.astype("int")
                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        time_end = datetime.datetime.now().timestamp()
        print("Time: ", time_end - time_start)
        
detector = Detector()
# detector.processImage("tasks/video_talkshow_low/frames/frame_5.jpg")
detector.processVideo("tracking/v2//video_talkshow_low.mp4")